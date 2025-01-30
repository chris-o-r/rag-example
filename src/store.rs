use pgvector::Vector;
use sqlx::{postgres::PgPoolOptions, Postgres};

use crate::Embedding;

const MAX_CONNECTIONS: u32 = 25;

pub async fn init(pool: &sqlx::Pool<sqlx::Postgres>) -> Result<(), anyhow::Error> {
    let _ = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;

    let _ = sqlx::query(
        "CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, embedding vector(768), document_text text, file_name text)",
    )
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn create_connection_pool(url: &str) -> Result<sqlx::Pool<Postgres>, anyhow::Error> {
    match PgPoolOptions::new()
        .max_connections(MAX_CONNECTIONS)
        .connect(url)
        .await
        .map_err(|err| {
            eprint!("Cannot connect to database [{}]", err.to_string());
            err
        }) {
        Ok(pool) => {
            println!("Connected to database successfully.");
            return Ok(pool);
        }
        Err(err) => {
            return Err(err.into());
        }
    }
}

pub async fn store_embeddings(
    pool: &sqlx::Pool<sqlx::Postgres>,
    embeddings: Vec<Embedding>,
    file_name: &str,
) -> Result<(), anyhow::Error> {
    let sql = r"INSERT INTO items (embedding, document_text, file_name) VALUES ($1, $2, $3)";

    for embedding in embeddings {
        let _ = sqlx::query(sql)
            .bind(embedding.vector_data)
            .bind(embedding.document_text)
            .bind(file_name)
            .execute(pool)
            .await?;
    }

    Ok(())
}

pub async fn retrieve_embeddings(
    pool: &sqlx::Pool<sqlx::Postgres>,
    embeddings: Vec<Vector>,
    num_embeddings: i32,
) -> Result<Vec<Embedding>, anyhow::Error> {
    let sql = r"SELECT * FROM items ORDER BY embedding <-> $1 LIMIT $2";

    sqlx::query_as::<_, Embedding>(sql)
        .bind(embeddings.get(0).unwrap())
        .bind(num_embeddings)
        .fetch_all(pool)
        .await
        .map_err(|err| err.into())
}

pub async fn truncate_table(pool: &sqlx::Pool<sqlx::Postgres>) -> Result<(), anyhow::Error> {
    let _ = sqlx::query("TRUNCATE TABLE items").execute(pool).await?;

    Ok(())
}
