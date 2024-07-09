use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
use clap::{crate_version, Arg, ArgAction, Command};
use endpoints::chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;

use wasmedge_wasi_nn as wasi_nn;

use wasi_nn::BackendError;

static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();

mod chat_ui {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum Role {
        #[serde(rename = "system")]
        System,
        #[serde(rename = "user")]
        User,
        #[serde(rename = "assistant")]
        Assistant,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Message {
        pub role: Role,
        pub content: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ChatBody {
        pub messages: Vec<Message>,
        #[serde(default)]
        pub channel_id: String,
    }

    pub enum TokenError {
        EndOfSequence = 1,
        ContextFull,
        PromptTooLong,
        TooLarge,
        InvalidEncoding,
        Other,
    }

    mod ffi {
        #[link(wasm_import_module = "chat_ui")]
        extern "C" {
            pub fn get_input(buf: *mut u8, buf_len: usize) -> usize;
            pub fn push_token(token_ptr: *const u8, token_len: usize) -> i32;
            pub fn return_token_error(error_code: i32);
        }
    }

    pub fn get_input() -> Result<ChatBody, serde_json::Error> {
        unsafe {
            let mut buf = [0u8; 1024];
            let mut s = Vec::<u8>::new();
            loop {
                let n = ffi::get_input(buf.as_mut_ptr(), buf.len());
                if n == 0 {
                    break;
                }
                s.extend_from_slice(&buf[0..n]);
            }
            serde_json::from_slice(&s)
        }
    }

    pub fn push_token(token: &str) -> bool {
        unsafe { ffi::push_token(token.as_ptr(), token.len()) >= 0 }
    }

    pub fn return_token_error(error: TokenError) {
        unsafe { ffi::return_token_error(error as i32) }
    }
}

impl chat_ui::ChatBody {
    fn to_message_request(self) -> ChatCompletionRequest {
        let mut chat_request = ChatCompletionRequest::default();
        for msg in self.messages {
            let role_fn = match msg.role {
                chat_ui::Role::System => {
                    ChatCompletionRequestMessage::new_system_message(msg.content, None)
                }
                chat_ui::Role::User => ChatCompletionRequestMessage::new_user_message(
                    endpoints::chat::ChatCompletionUserMessageContent::Text(msg.content),
                    None,
                ),
                chat_ui::Role::Assistant => ChatCompletionRequestMessage::new_assistant_message(
                    Some(msg.content),
                    None,
                    None,
                ),
            };

            chat_request.messages.push(role_fn);
        }
        chat_request
    }
}

#[allow(unreachable_code)]
fn main() -> Result<(), String> {
    let matches = Command::new("llama-chat")
        .version(crate_version!())
        .arg(
            Arg::new("model_alias")
                .short('a')
                .long("model-alias")
                .value_name("ALIAS")
                .help("Model alias")
                .default_value("default"),
        )
        .arg(
            Arg::new("ctx_size")
                .short('c')
                .long("ctx-size")
                .value_parser(clap::value_parser!(u64))
                .value_name("CTX_SIZE")
                .help("Size of the prompt context")
                .default_value("512"),
        )
        .arg(
            Arg::new("n_predict")
                .short('n')
                .long("n-predict")
                .value_parser(clap::value_parser!(u64))
                .value_name("N_PRDICT")
                .help("Number of tokens to predict")
                .default_value("1024"),
        )
        .arg(
            Arg::new("n_gpu_layers")
                .short('g')
                .long("n-gpu-layers")
                .value_parser(clap::value_parser!(u64))
                .value_name("N_GPU_LAYERS")
                .help("Number of layers to run on the GPU")
                .default_value("100"),
        )
        .arg(
            Arg::new("batch_size")
                .short('b')
                .long("batch-size")
                .value_parser(clap::value_parser!(u64))
                .value_name("BATCH_SIZE")
                .help("Batch size for prompt processing")
                .default_value("512"),
        )
        .arg(
            Arg::new("temp")
                .long("temp")
                .value_parser(clap::value_parser!(f32))
                .value_name("TEMP")
                .help("Temperature for sampling")
                .default_value("0.8"),
        )
        .arg(
            Arg::new("repeat_penalty")
                .long("repeat-penalty")
                .value_parser(clap::value_parser!(f32))
                .value_name("REPEAT_PENALTY")
                .help("Penalize repeat sequence of tokens")
                .default_value("1.1"),
        )
        .arg(
            Arg::new("reverse_prompt")
                .short('r')
                .long("reverse-prompt")
                .value_name("REVERSE_PROMPT")
                .help("Halt generation at PROMPT, return control."),
        )
        .arg(
            Arg::new("system_prompt")
                .short('s')
                .long("system-prompt")
                .value_name("SYSTEM_PROMPT")
                .help("System prompt message string")
                .default_value("[Default system message for the prompt template]"),
        )
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "llama-3-chat",
                    "codellama-instruct",
                    "codellama-super-instruct",
                    "mistral-instruct",
                    "mistrallite",
                    "openchat",
                    "human-assistant",
                    "vicuna-1.0-chat",
                    "vicuna-1.1-chat",
                    "chatml",
                    "baichuan-2",
                    "wizard-coder",
                    "zephyr",
                    "stablelm-zephyr",
                    "intel-neural",
                    "deepseek-chat",
                    "deepseek-coder",
                    "solar-instruct",
                    "phi-2-instruct",
                    "phi-3-chat",
                    "phi-3-instruct",
                    "gemma-instruct",
                ])
                .value_name("TEMPLATE")
                .help("Prompt template.")
                .default_value("llama-2-chat"),
        )
        .arg(
            Arg::new("log_prompts")
                .long("log-prompts")
                .value_name("LOG_PROMPTS")
                .help("Print prompt strings to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_stat")
                .long("log-stat")
                .value_name("LOG_STAT")
                .help("Print statistics to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_all")
                .long("log-all")
                .value_name("LOG_all")
                .help("Print all log information to stdout")
                .action(ArgAction::SetTrue),
        )
        .after_help("Example: the command to run `llama-2-7B` model,\n  wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat\n")
        .get_matches();

    // create an `Options` instance
    let mut options = Options::default();

    // model alias
    let model_name = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[INFO] Model alias: {alias}", alias = &model_name);

    // prompt context size
    let ctx_size = matches.get_one::<u64>("ctx_size").unwrap();
    println!("[INFO] Prompt context size: {size}", size = ctx_size);
    options.ctx_size = *ctx_size;

    // max buffer size
    if MAX_BUFFER_SIZE.set(*ctx_size as usize).is_err() {
        return Err(String::from(
            "Fail to set `MAX_BUFFER_SIZE`. It is already set.",
        ));
    }

    // number of tokens to predict
    let n_predict = matches.get_one::<u64>("n_predict").unwrap();
    println!("[INFO] Number of tokens to predict: {n}", n = n_predict);
    options.n_predict = *n_predict;

    // n_gpu_layers
    let n_gpu_layers = matches.get_one::<u64>("n_gpu_layers").unwrap();
    println!(
        "[INFO] Number of layers to run on the GPU: {n}",
        n = n_gpu_layers
    );
    options.n_gpu_layers = *n_gpu_layers;

    // batch size
    let batch_size = matches.get_one::<u64>("batch_size").unwrap();
    println!(
        "[INFO] Batch size for prompt processing: {size}",
        size = batch_size
    );
    options.batch_size = *batch_size;

    // temperature
    let temp = matches.get_one::<f32>("temp").unwrap();
    println!("[INFO] Temperature for sampling: {temp}", temp = temp);
    options.temp = *temp;

    // repeat penalty
    let repeat_penalty = matches.get_one::<f32>("repeat_penalty").unwrap();
    println!(
        "[INFO] Penalize repeat sequence of tokens: {penalty}",
        penalty = repeat_penalty
    );
    options.repeat_penalty = *repeat_penalty;

    // reverse_prompt
    if let Some(reverse_prompt) = matches.get_one::<String>("reverse_prompt") {
        println!("[INFO] Reverse prompt: {prompt}", prompt = &reverse_prompt);
        options.reverse_prompt = Some(reverse_prompt.to_string());
    }

    // type of prompt template
    let prompt_template = matches
        .get_one::<String>("prompt_template")
        .unwrap()
        .to_string();
    let template_ty = match PromptTemplateType::from_str(&prompt_template) {
        Ok(template) => template,
        Err(e) => {
            return Err(format!(
                "Fail to parse prompt template type: {msg}",
                msg = e.to_string()
            ))
        }
    };
    println!("[INFO] Prompt template: {ty:?}", ty = &template_ty);

    // log prompts
    let log_prompts = matches.get_flag("log_prompts");
    println!("[INFO] Log prompts: {enable}", enable = log_prompts);

    // log statistics
    let log_stat = matches.get_flag("log_stat");
    println!("[INFO] Log statistics: {enable}", enable = log_stat);

    // log all
    let log_all = matches.get_flag("log_all");
    println!("[INFO] Log all information: {enable}", enable = log_all);

    // set `log_enable`
    if log_stat || log_all {
        options.log_enable = true;
    }

    let template = create_prompt_template(template_ty.clone());

    // serialize metadata
    let metadata = match serde_json::to_string(&options) {
        Ok(metadata) => metadata,
        Err(e) => {
            return Err(format!(
                "Fail to serialize options: {msg}",
                msg = e.to_string()
            ))
        }
    };

    if log_stat || log_all {
        print_log_begin_separator(
            "MODEL INFO (Load Model & Init Execution Context)",
            Some("*"),
            None,
        );
    }

    // load the model into wasi-nn
    let graph = match wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Ggml,
        wasi_nn::ExecutionTarget::AUTO,
    )
    .config(metadata)
    .build_from_cache(model_name.as_ref())
    {
        Ok(graph) => graph,
        Err(e) => {
            return Err(format!(
                "Fail to load model into wasi-nn: {msg}",
                msg = e.to_string()
            ))
        }
    };

    // initialize the execution context
    let mut context = match graph.init_execution_context() {
        Ok(context) => context,
        Err(e) => {
            return Err(format!(
                "Fail to create wasi-nn execution context: {msg}",
                msg = e.to_string()
            ))
        }
    };

    if log_stat || log_all {
        print_log_end_separator(Some("*"), None);
    }

    loop {
        let mut chat_request = match chat_ui::get_input() {
            Ok(r) => r.to_message_request(),
            Err(_) => continue,
        };

        if log_stat || log_all {
            print_log_begin_separator("STATISTICS (Set Input)", Some("*"), None);
        }

        // build prompt
        let max_prompt_tokens = *ctx_size * 4 / 5;
        let prompt = match build_prompt(
            &template,
            &mut chat_request,
            &mut context,
            max_prompt_tokens,
        ) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(format!(
                    "Fail to generate prompt. Reason: {msg}",
                    msg = e.to_string()
                ))
            }
        };

        if log_stat || log_all {
            print_log_end_separator(Some("*"), None);
        }

        if log_prompts || log_all {
            print_log_begin_separator("PROMPT", Some("*"), None);
            println!("{}", &prompt);
            print_log_end_separator(Some("*"), None);
        }

        if log_stat || log_all {
            print_log_begin_separator("STATISTICS (Compute)", Some("*"), None);
        }

        // compute
        let result = match options.reverse_prompt {
            Some(ref reverse_prompt) => stream_compute(&mut context, Some(reverse_prompt.as_str())),
            None => stream_compute(&mut context, None),
        };

        if log_stat || log_all {
            print_log_end_separator(Some("*"), None);
        }

        match result {
            Ok(_) => {
                chat_ui::return_token_error(chat_ui::TokenError::EndOfSequence);
            }
            Err(wasi_nn::Error::BackendError(BackendError::EndOfSequence)) => {
                chat_ui::return_token_error(chat_ui::TokenError::EndOfSequence);
            }
            Err(wasi_nn::Error::BackendError(BackendError::ContextFull)) => {
                chat_ui::return_token_error(chat_ui::TokenError::ContextFull);
            }
            Err(wasi_nn::Error::BackendError(BackendError::PromptTooLong)) => {
                chat_ui::return_token_error(chat_ui::TokenError::PromptTooLong);
            }
            Err(wasi_nn::Error::BackendError(BackendError::TooLarge)) => {
                chat_ui::return_token_error(chat_ui::TokenError::TooLarge);
            }
            Err(wasi_nn::Error::BackendError(BackendError::InvalidEncoding)) => {
                chat_ui::return_token_error(chat_ui::TokenError::InvalidEncoding);
            }
            Err(e) => {
                eprintln!(
                    "[chat_ui.wasm] Fail to compute. Reason: {msg}",
                    msg = e.to_string()
                );
                chat_ui::return_token_error(chat_ui::TokenError::Other);
            }
        }
        context.fini_single().unwrap();
    }

    Ok(())
}

fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>,
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str("\n");
    println!("{}", separator);
    total_len
}

fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n");
    println!("{}", separator);
}

fn create_prompt_template(template_ty: PromptTemplateType) -> ChatPrompt {
    match template_ty {
        PromptTemplateType::Llama2Chat => {
            ChatPrompt::Llama2ChatPrompt(chat_prompts::chat::llama::Llama2ChatPrompt)
        }
        PromptTemplateType::Llama3Chat => {
            ChatPrompt::Llama3ChatPrompt(chat_prompts::chat::llama::Llama3ChatPrompt)
        }
        PromptTemplateType::MistralInstruct => {
            ChatPrompt::MistralInstructPrompt(chat_prompts::chat::mistral::MistralInstructPrompt)
        }
        PromptTemplateType::MistralLite => {
            ChatPrompt::MistralLitePrompt(chat_prompts::chat::mistral::MistralLitePrompt)
        }
        PromptTemplateType::OpenChat => {
            ChatPrompt::OpenChatPrompt(chat_prompts::chat::openchat::OpenChatPrompt)
        }
        PromptTemplateType::CodeLlama => {
            ChatPrompt::CodeLlamaInstructPrompt(chat_prompts::chat::llama::CodeLlamaInstructPrompt)
        }
        PromptTemplateType::CodeLlamaSuper => ChatPrompt::CodeLlamaSuperInstructPrompt(
            chat_prompts::chat::llama::CodeLlamaSuperInstructPrompt,
        ),
        PromptTemplateType::HumanAssistant => ChatPrompt::HumanAssistantChatPrompt(
            chat_prompts::chat::belle::HumanAssistantChatPrompt,
        ),
        PromptTemplateType::VicunaChat => {
            ChatPrompt::VicunaChatPrompt(chat_prompts::chat::vicuna::VicunaChatPrompt)
        }
        PromptTemplateType::Vicuna11Chat => {
            ChatPrompt::Vicuna11ChatPrompt(chat_prompts::chat::vicuna::Vicuna11ChatPrompt)
        }
        PromptTemplateType::VicunaLlava => {
            ChatPrompt::VicunaLlavaPrompt(chat_prompts::chat::vicuna::VicunaLlavaPrompt)
        }
        PromptTemplateType::ChatML => {
            ChatPrompt::ChatMLPrompt(chat_prompts::chat::chatml::ChatMLPrompt)
        }
        PromptTemplateType::Baichuan2 => {
            ChatPrompt::Baichuan2ChatPrompt(chat_prompts::chat::baichuan::Baichuan2ChatPrompt)
        }
        PromptTemplateType::WizardCoder => {
            ChatPrompt::WizardCoderPrompt(chat_prompts::chat::wizard::WizardCoderPrompt)
        }
        PromptTemplateType::Zephyr => {
            ChatPrompt::ZephyrChatPrompt(chat_prompts::chat::zephyr::ZephyrChatPrompt)
        }
        PromptTemplateType::StableLMZephyr => ChatPrompt::StableLMZephyrChatPrompt(
            chat_prompts::chat::zephyr::StableLMZephyrChatPrompt,
        ),
        PromptTemplateType::IntelNeural => {
            ChatPrompt::NeuralChatPrompt(chat_prompts::chat::intel::NeuralChatPrompt)
        }
        PromptTemplateType::DeepseekChat => {
            ChatPrompt::DeepseekChatPrompt(chat_prompts::chat::deepseek::DeepseekChatPrompt)
        }
        PromptTemplateType::DeepseekCoder => {
            ChatPrompt::DeepseekCoderPrompt(chat_prompts::chat::deepseek::DeepseekCoderPrompt)
        }
        PromptTemplateType::SolarInstruct => {
            ChatPrompt::SolarInstructPrompt(chat_prompts::chat::solar::SolarInstructPrompt)
        }
        PromptTemplateType::Phi2Chat => {
            ChatPrompt::Phi2ChatPrompt(chat_prompts::chat::phi::Phi2ChatPrompt)
        }
        PromptTemplateType::Phi2Instruct => {
            ChatPrompt::Phi2InstructPrompt(chat_prompts::chat::phi::Phi2InstructPrompt)
        }
        PromptTemplateType::Phi3Chat => {
            ChatPrompt::Phi3ChatPrompt(chat_prompts::chat::phi::Phi3ChatPrompt)
        }
        PromptTemplateType::Phi3Instruct => {
            ChatPrompt::Phi3InstructPrompt(chat_prompts::chat::phi::Phi3InstructPrompt)
        }
        PromptTemplateType::GemmaInstruct => {
            ChatPrompt::GemmaInstructPrompt(chat_prompts::chat::gemma::GemmaInstructPrompt)
        }
        PromptTemplateType::Octopus => {
            ChatPrompt::OctopusPrompt(chat_prompts::chat::octopus::OctopusPrompt)
        }
        _ => {
            panic!("Unsupported prompt template type: {:?}", template_ty)
        }
    }
}

fn stream_compute(
    context: &mut wasi_nn::GraphExecutionContext,
    stop: Option<&str>,
) -> Result<(), wasi_nn::Error> {
    // Compute one token at a time, and get the token using the get_output_single().
    // Retrieve the output.
    let max_output_size = *MAX_BUFFER_SIZE.get().unwrap();
    let mut output_buffer = vec![0u8; max_output_size];
    let mut start_offset = 0;

    loop {
        context.compute_single()?;

        let mut output_size = context
            .get_output_single(0, &mut output_buffer[start_offset..])
            .unwrap();
        output_size = start_offset + std::cmp::min(max_output_size, output_size);

        if std::str::from_utf8(&output_buffer[..output_size]).is_err() {
            start_offset = output_size;
            continue;
        }

        start_offset = 0;

        let token = String::from_utf8(output_buffer[..output_size].to_vec()).unwrap();

        if let Some(stop) = stop {
            if token == stop {
                break;
            }
        }
        if !chat_ui::push_token(&token) {
            return Ok(());
        }
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct Options {
    #[serde(rename = "enable-log")]
    log_enable: bool,
    #[serde(rename = "ctx-size")]
    ctx_size: u64,
    #[serde(rename = "n-predict")]
    n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    batch_size: u64,
    #[serde(rename = "temp")]
    temp: f32,
    #[serde(rename = "repeat-penalty")]
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}

fn build_prompt(
    template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
    context: &mut wasi_nn::GraphExecutionContext,
    max_prompt_tokens: u64,
) -> Result<String, String> {
    loop {
        // build prompt
        let prompt = match template.build(&mut chat_request.messages) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(format!(
                    "Fail to build chat prompts: {msg}",
                    msg = e.to_string()
                ))
            }
        };

        // read input tensor
        let tensor_data = prompt.trim().as_bytes().to_vec();
        if context
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .is_err()
        {
            return Err(String::from("Fail to set input tensor"));
        };

        // Retrieve the number of prompt tokens.
        let max_input_size = *MAX_BUFFER_SIZE.get().unwrap();
        let mut input_buffer = vec![0u8; max_input_size];
        let mut input_size = context.get_output(1, &mut input_buffer).unwrap();
        input_size = std::cmp::min(max_input_size, input_size);
        let token_info: Value = serde_json::from_slice(&input_buffer[..input_size]).unwrap();
        let prompt_tokens = token_info["input_tokens"].as_u64().unwrap();

        match prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role() {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() >= 4 {
                            if chat_request.messages[1].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                            if chat_request.messages[1].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(1);
                            }
                        } else if chat_request.messages.len() == 3
                            && chat_request.messages[1].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(1);
                        } else {
                            return Ok(prompt);
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() >= 3 {
                            if chat_request.messages[0].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(0);
                            }
                            if chat_request.messages[0].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(0);
                            }
                        } else if chat_request.messages.len() == 2
                            && chat_request.messages[0].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(0);
                        } else {
                            return Ok(prompt);
                        }
                    }
                    _ => panic!("Found a unsupported chat message role!"),
                }
                continue;
            }
            false => return Ok(prompt),
        }
    }
}
