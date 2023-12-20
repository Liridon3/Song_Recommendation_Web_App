import os
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, jsonify, flash
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

application.secret_key = os.urandom(24)

# Load the dataset
data = pd.read_csv('spotify_songs.csv')

# Drop unnecessary columns for clustering
data_for_clustering = data.drop(
    ['track_name', 'track_artist', 'track_album_id', 'track_album_name',
     'track_album_release_date', 'playlist_name', 'playlist_id', 'duration_ms'], axis=1)

# One-hot encode genre and subgenre columns
data_for_clustering = pd.get_dummies(data_for_clustering,
                                     columns=['playlist_genre', 'playlist_subgenre'])
# Save the 'track_id' for later use
track_ids = data_for_clustering['track_id']

# Drop 'track_id' before scaling
data_for_clustering = data_for_clustering.drop('track_id', axis=1)

# Standardize the numeric data
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# Create a KMeans clustering model with 24 clusters
kmeans = KMeans(n_clusters=24, random_state=42, n_init=10)

# Fit the model and predict cluster labels for each data point
data['cluster'] = kmeans.fit_predict(data_for_clustering_scaled)

# Calculate Silhouette Score for the clustering
silhouette_avg = silhouette_score(data_for_clustering_scaled, data['cluster'])
print(f"Silhouette Score for Clustering: {silhouette_avg}")

# Apply PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
data_for_clustering_pca = pca.fit_transform(data_for_clustering_scaled)

# Davies-Bouldin Index visualization
fig_davies_bouldin = px.scatter(
    data_for_clustering_pca,
    x=data_for_clustering_pca[:, 0],
    y=data_for_clustering_pca[:, 1],
    color=data['cluster'],
    color_discrete_sequence=px.colors.qualitative.Set1,
    labels={'0': 'Principal Component 1', '1': 'Principal Component 2'},
    title='Davies-Bouldin Index Visualization'
)
fig_davies_bouldin.update_layout(paper_bgcolor='black', font_color='white', plot_bgcolor='black')
fig_davies_bouldin_html = fig_davies_bouldin.to_html(full_html=False)
fig_davies_bouldin.write_html("templates/davies_bouldin_visualization.html")

# WSS for different values of k
wss_values = []
k_values = range(2, 30)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_for_clustering_scaled)
    wss_values.append(kmeans.inertia_)
fig_wss = px.line(x=k_values, y=wss_values, title='Total Within-Cluster Sum of Squares (WSS)',
                  labels={'x': 'Number of Clusters (k)', 'y': 'WSS'})
fig_wss.update_layout(paper_bgcolor='black', font_color='white', plot_bgcolor='black')
fig_wss.write_html("templates/wss_plot.html")

# average popularity by genre visualization
genre_popularity = data.groupby('playlist_genre')['track_popularity'].mean().reset_index()
fig_genre_popularity = px.bar(genre_popularity, x='playlist_genre', y='track_popularity',
                              title='Average Genre Popularity',
                              labels={'track_popularity': 'Average Popularity', 'playlist_genre': 'Genre'},
                              color='playlist_genre', color_discrete_sequence=px.colors.qualitative.Set1)
fig_genre_popularity.update_layout(paper_bgcolor='black', font_color='white', plot_bgcolor='black')
fig_genre_popularity.write_html("templates/genre_popularity.html")

# the top 10 most popular songs visualization
top_popular_songs = data.nlargest(10, 'track_popularity')
fig_popular_songs = px.bar(top_popular_songs, x='track_name', y='track_popularity', title='Top 10 Most Popular Songs')
fig_popular_songs.update_layout(paper_bgcolor='black', font_color='white', plot_bgcolor='black')
fig_popular_songs.update_layout(xaxis_title='Track Name', yaxis_title='Popularity')
fig_popular_songs.write_html("templates/top_popular_songs.html")


# Route to handle autocomplete requests
@application.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term', '').lower()
    matching_songs = data[data['track_name'].str.lower().str.contains(term, na=False)]
    autocomplete_data = []

    for index, row in matching_songs.iterrows():
        artists = [artist.strip() for artist in row['track_artist'].split(',')]
        artist_str = ', '.join(artists)
        autocomplete_data.append({'label': f"{row['track_name']} - {artist_str}", 'value': row['track_id']})

    return jsonify(autocomplete_data)


# Route to render Davies Bouldin visualization
@application.route('/davies_bouldin_visualization', methods=['GET'])
def davies_bouldin_visualization():
    return render_template('davies_bouldin_visualization.html')


# Route to render the within-cluster sum of squares
@application.route('/wss_plot', methods=['GET'])
def wss_visualization():
    return render_template('templates/wss_plot.html')


# Route to render the most popular songs
@application.route('/top_popular_songs', methods=['GET'])
def top_popular_songs():
    return render_template('top_popular_songs.html')


# Route to render the popularity by genre
@application.route('/genre_popularity', methods=['GET'])
def top_popular_artists():
    return render_template('genre_popularity.html')


# Main route for rendering the index page
@application.route('/', methods=['GET', 'POST'])
def index():
    user_song = None
    recommended_songs = pd.DataFrame()

    if request.method == 'POST':
        # Get the selected track ID from the form
        selected_track_id = request.form['selected_track_id']

        # Check if the selected track ID is empty or not a valid track ID
        if not selected_track_id or selected_track_id not in data['track_id'].values:
            flash('Please select a valid song from the drop-down.', 'error')
            return render_template('index.html', user_song=user_song, recommended_songs=recommended_songs)

        # Find the selected song in the dataset using the track ID
        user_song = data[data['track_id'] == selected_track_id].iloc[0]

        # Get the cluster of the user-entered song
        user_cluster = user_song['cluster']

        # Filter songs in the same cluster as the user-entered song
        cluster_songs = data[data['cluster'] == user_cluster]

        # Exclude the user-entered song from recommendations
        cluster_songs = cluster_songs[cluster_songs['track_id'] != user_song['track_id']]

        # Get data ready to calculate cosine similarity between the user song and cluster songs
        data_for_clustering['track_id'] = track_ids
        user_song_cluster = data.loc[data['track_id'] == selected_track_id, 'cluster'].values[0]
        cluster_songs_data = data[data['cluster'] == user_song_cluster].index

        # Extract the corresponding rows from the scaled data
        user_song_data = data_for_clustering_scaled[data['track_id'] == selected_track_id]
        user_cluster = data.loc[data['track_id'] == selected_track_id, 'cluster'].values[0]
        cluster_songs_data = data_for_clustering_scaled[data['cluster'] == user_cluster]

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(user_song_data, cluster_songs_data)

        # Get the indices of songs sorted by similarity in descending order
        sorted_indices = similarity_scores.argsort()[0][::-1]

        # Select the top 10 most similar songs
        recommended_songs = cluster_songs.iloc[sorted_indices[:10]]

    return render_template('index.html', user_song=user_song, recommended_songs=recommended_songs)


if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 8000))
    #application.run(host='0.0.0.0', port=port)
    application.run()
