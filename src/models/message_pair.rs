use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MessagePair {
    pub question: String,
    pub answer: String,
    pub references: Vec<String>,
}

impl MessagePair {
    pub fn new(question: String, answer: String, references: Vec<String>) -> Self {
        MessagePair {
            question,
            answer,
            references,
        }
    }
}
