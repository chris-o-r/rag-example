use dotenv::dotenv;
use std::{env, fs};

use text_splitter::{ChunkConfig, TextSplitter};
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tokenizers::Tokenizer;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use models::{embedding::Embedding, message_pair::MessagePair};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{self, ChatCompletionRequest, ChatCompletionResponse},
    common::GPT4_O_MINI,
};
use pgvector::Vector;

mod character_text_splitter;
mod models;
mod store;

const CREATE_EMBEDDINGS: bool = true;

const QUESTIONS: [&str; 11] = ["What are the main objectives of the study on Table Information Seeking (TIS) in Large Language Models (LLMs)?", "How does the newly introduced benchmark, TabIS, differ from previous evaluation methods for table information extraction?, ", "What were the performance results of various LLMs tested using the TabIS benchmark, particularly regarding their understanding of table structures?
", "Who are the authors of the paper?", "What is the title of the paper?", "What is the abstract of the paper?", "What is the introduction of the paper?", "What is the methodology of the paper?", "What is the related work of the paper?", "What is the conclusion of the paper?", "What is the future work of the paper?"];

const DOCUMENT: &str = "assets/research.txt";

#[tokio::main]
async fn main() {
    dotenv().ok();

    let pool =
        store::create_connection_pool("postgres://postgres:password@127.0.0.1:5432/example_db")
            .await
            .map_err(|err| {
                panic!("Cannot connect to database [{}]", err.to_string());
            })
            .unwrap();

    let _ = store::init(&pool).await;

    // With custom InitOptions
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Base).with_show_download_progress(true),
    )
    .unwrap();

    if CREATE_EMBEDDINGS {
        let _ = store::truncate_table(&pool).await;

        let vec_text_pairs = create_embeddings_from_document(&model, DOCUMENT)
            .map_err(|error| {
                panic!(
                    "Cannot create embeddings from document [{}]",
                    error.to_string()
                );
            })
            .unwrap();

        match store::store_embeddings(&pool, vec_text_pairs, DOCUMENT).await {
            Ok(_) => {
                println!("Stored embeddings successfully.");
            }
            Err(err) => {
                eprint!("Cannot store embeddings [{}]", err.to_string());
                panic!()
            }
        };
    }

    let mut results: Vec<MessagePair> = Vec::new();

    let system_prompt = fs::read_to_string("assets/system-prompt.txt").unwrap();

    let mut chat_window = String::new();

    chat_window.push_str(&system_prompt);

    for question in QUESTIONS {
        let question_embeddings = model
            .embed(vec![question.to_lowercase()], None)
            .unwrap()
            .iter()
            .map(|x| Vector::from(x.clone()))
            .collect::<Vec<Vector>>();

        let embeddings: Vec<Embedding> = store::retrieve_embeddings(&pool, question_embeddings, 3)
            .await
            .map_err(|err| {
                panic!("Cannot retrieve embeddings [{}]", err.to_string());
            })
            .unwrap();

        let mut index = 0;
        let references_text = embeddings
            .iter()
            .map(|embedding| embedding.document_text.clone())
            .reduce(|acc, x| {
                index += 1;
                acc + &format!("\n\n{}. {}", index, x)
            })
            .unwrap_or_else(|| String::new());

        chat_window.push_str(
            format!(
                "---Begin Question\n{}\n--End Question --Begin Context\n{}\n--End Context",
                question, references_text
            )
            .as_str(),
        );

        let message = match make_open_api_request(&chat_window).await {
            Ok(response) => response.choices[0]
                .message
                .content
                .as_ref()
                .unwrap()
                .clone(),
            Err(err) => {
                format!("Cannot make OpenAI request [{}]", err.to_string())
            }
        };

        chat_window.push_str(&format!("\n\n{}", message));

        results.push(MessagePair::new(
            question.to_string(),
            message,
            embeddings
                .iter()
                .map(|embedding| embedding.document_text.clone())
                .collect(),
        ));
    }

    let json = serde_json::to_string(&results).unwrap();

    fs::write("results.json", json).unwrap();

    // Print the results as json
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

async fn make_open_api_request(message: &str) -> Result<ChatCompletionResponse, anyhow::Error> {
    let api_key = env::var("OPENAI_API_KEY").unwrap().to_string();
    let client = OpenAIClient::builder()
        .with_api_key(api_key)
        .build()
        .map_err(|err| {
            anyhow::Error::msg(format!("Cannot create OpenAI client [{}]", err.to_string()))
        })?;

    let req = ChatCompletionRequest::new(
        GPT4_O_MINI.to_string(),
        vec![chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::user,
            content: chat_completion::Content::Text(String::from(message)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
    );

    client.chat_completion(req).await.map_err(|err| err.into())
}

fn create_embeddings_from_document(
    model: &TextEmbedding,
    document_name: &str,
) -> Result<Vec<Embedding>, anyhow::Error> {
    let document_text: String = std::fs::read_to_string(document_name)?;

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let max_tokens = 1000;
    let splitter: TextSplitter<Tokenizer> =
        TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    let document_chunks = splitter.chunks(&document_text).collect::<Vec<_>>();

    let embeddings: Vec<Vector> = model
        .embed(document_chunks.clone(), None)
        .unwrap()
        .iter()
        .map(|x| Vector::from(x.clone()))
        .collect::<Vec<Vector>>();

    let vec_text_pairs: Vec<Embedding> = document_chunks
        .iter()
        .cloned()
        .zip(embeddings.iter().cloned())
        .map(|(text, embedding)| Embedding {
            document_text: text.to_string(),
            vector_data: embedding,
            file_name: document_name.to_string(),
        })
        .collect();

    Ok(vec_text_pairs)
}
