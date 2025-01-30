use sqlx::{postgres::PgRow, Row};


pub struct Embedding {
    pub document_text: String,
    pub vector_data: pgvector::Vector,
    pub file_name: String,
}

impl<'r> sqlx::FromRow<'r, PgRow> for Embedding {
    fn from_row(row: &'r PgRow) -> Result<Self, sqlx::Error> {
        Ok(Embedding {
            document_text: row.try_get("document_text")?,
            vector_data: row.try_get("embedding")?,
            file_name: row.try_get("file_name")?,
        })
    }
}
