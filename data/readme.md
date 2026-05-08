This folder contains vector embeddings of texts (transcripts of chosen Youtube videos).

The models used are: 
openai/text-embedding-3-small ('openai')
SamLowe/roberta-base-go_emotions ('go_emotion')

The files contain embeddings of video raw transcripts. 

`golden` refers to the 51 successfully transcript-fetched videos introduced in a previous work led by Prof. Larry Liebovitch.
`add00` refers to additional 36 videos with manually assigned 6 classes.

All video transcripts are tokenized and chunked into 200-token chunks and then embedded. The `per_video` is an exceptional file because each video has less than 8192 tokens which can be fully fed into the openai embedding model.

The fields are:
`video_id`: Same as described in the `Transcripts` folder.
`chunk_id`: Indicates the order of the chunked text in the original raw texts.
`text`: The texts of the chunk.
`total_tokens`: The total number of tokens in the chunk.
`tokens`: The array of tokens, obtained from tokenizing the chunk raw text, fully dependent on the embedding model chosen.
`embedding`: The full embedded vector, dependent on the chosen embedding model.
`class`: Manually assigned classes.
`class_label`: A unique number mapped from the `class`.

Here is an example code to load the embedded vectors in python:
```
import pandas as pd
import numpy as np
import pickle

file = "VectorEmbeddings/openai_embedding_golden.pkl"
with open(file, "rb") as f:
    loaded_data = pickle.load(f)

# The dataframe
df = pd.DataFrame(loaded_data)

print(f"Loaded {len(df)} records.")
print(f"Dimension: {len(df['embedding'].iloc[0])}")

# The embedding matrix
X = np.array(df['embedding'].to_list())
```