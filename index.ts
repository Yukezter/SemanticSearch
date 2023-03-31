import { Configuration, OpenAIApi } from 'openai'
import express from 'express'

const cosine_similarity = (vector1: number[], vector2: number[]) => {
  let dot = 0, norm1 = 0, norm2 = 0;

  for (let i = 0; i < vector1.length; i++) {
    dot += vector1[i] * vector2[i];
    norm1 += vector1[i] * vector1[i];
    norm2 += vector2[i] * vector2[i];
  }

  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  return dot / (norm1 * norm2);
}

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

const app = express();

app.get('/groups/create', async (req, res) => {
  const { name, description } = req.body
  // https://platform.openai.com/docs/api-reference/embeddings/create
  const { data } = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: description,
  });

  const { embedding } = data.data[0];
  const group = { name, description, embedding };

  res.status(200).json(group);
})

app.get('/search', async (req, res) => {
  const { search } = req.query; // ?search=string
  const { data } = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: search as string,
  });

  const { embedding } = data.data[0];
  // https://redis.io/docs/stack/search/quick_start/
  // https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases/redis
  const db_embeddings = [[1, 2, 3]]; // here we query all embeddings from db

  // calculate similarity values and sort in descending order
  // the closer to 1 the result is, the more similar it is
  // maybe just get the first 10 results
  const smiliarities = db_embeddings.map(vector => {
    return cosine_similarity(vector, embedding)
  }).sort((a, b) => b - a).slice(0, 10);

  res.json(smiliarities);
})

const PORT = 8000;
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});

