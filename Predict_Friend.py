import PySimpleGUI as sg
from gensim.models import Word2Vec

# Load the trained Word2Vec model
model = Word2Vec.load('twitter_embeddings.model')


# Function to find potential friends for a given user
def find_potential_friends(user_id, top_n=5):
    try:
        # Retrieve the embedding for the given user
        user_embedding = model.wv[user_id]

        # Find the most similar users (potential friends)
        similar_users = model.wv.most_similar([user_embedding], topn=top_n)

        # Extract user IDs of potential friends
        potential_friends = [user for user, _ in similar_users]

        return potential_friends
    except KeyError:
        return []


# Define the GUI layout
layout = [
    [sg.Text("Enter User ID:")],
    [sg.InputText(key="USER_ID")],
    [sg.Button("Find Potential Friends")],
    [sg.Text("", size=(30, 5), key="FRIENDS_TEXT")],
    [sg.Button("Terminate", button_color=('white', 'red')), sg.Button("Re-attempt")],
]
sg.theme('BluePurple')
# Set window size and background color, and create the window
window = sg.Window("Twitter Friend Suggestion", layout, size=(400, 300))  # Adjusted size and background color

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Terminate":
        break

    if event == "Find Potential Friends":
        user_id_to_predict = values["USER_ID"]
        potential_friends = find_potential_friends(user_id_to_predict)

        if potential_friends:
            friends_text = f"Potential friends for user {user_id_to_predict}:\n"
            friends_text += "\n".join(potential_friends)
        else:
            friends_text = f"No potential friends found for user {user_id_to_predict}."

        window["FRIENDS_TEXT"].update(friends_text)

    if event == "Re-attempt":
        window["USER_ID"].update("")  # Clear the input field
        window["FRIENDS_TEXT"].update("")  # Clear the previous results

# Close the window
window.close()
