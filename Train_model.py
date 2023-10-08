import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv", sep=',(?=\S)', engine='python')

# Split the data into training and testing sets (60% for training, 40% for testing)
train_df, test_df = train_test_split(df, test_size=0.8, random_state=42)

# Create a directed graph using NetworkX
G = nx.DiGraph()

# Add nodes (Twitter users) for the training data
for index, row in train_df.iterrows():
    G.add_node(row['id'], screenName=row['screenName'], tags=row['tags'],
               avatar=row['avatar'], followersCount=row['followersCount'],
               friendsCount=row['friendsCount'], lang=row['lang'],
               lastSeen=row['lastSeen'], tweetId=row['tweetId'])

# Add edges (connections between Twitter users) for the training data
for index, row in train_df.iterrows():
    # Clean the 'friends' data by removing double quotes and extra spaces
    cleaned_friends = [friend.strip('" ') for friend in row['friends'].split(',')]
    for friend in cleaned_friends:
        friend_id = friend.strip()  # Remove any remaining whitespace
        if friend_id.isdigit():  # Check if it's a numeric value
            G.add_edge(row['id'], int(friend_id))  # Convert to integer and add as an edge

# Define parameters for random walk generation
num_walks_per_node = 10
walk_length = 10

# Build the vocabulary using the training data
sentences = []
for node in G.nodes:
    for _ in range(num_walks_per_node):
        walk = [str(node)]
        current_node = node
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                next_node = random.choice(neighbors)
                walk.append(str(next_node))
                current_node = next_node
            else:
                break
        sentences.append(walk)

# Initialize the Word2Vec model and build the vocabulary
model = Word2Vec(vector_size=64, window=5, sg=1, workers=4)
model.build_vocab(sentences)

# Train the Word2Vec model using the training data
model.train(sentences, total_examples=len(sentences), epochs=10)  # Adjust the number of epochs as needed

# Save the embeddings to a file
model.save('twitter_embeddings.model')


