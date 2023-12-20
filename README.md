
# Flask Song Recommendation App

## Introduction
This Flask application provides a song recommendation feature using Spotify song data. It leverages machine learning techniques, including KMeans clustering and cosine similarity, to offer song suggestions based on user preferences.

## Setup Instructions

### Prerequisites
- Python 3.x installed
- Pip (Python package manager)

### Installation Steps

1. Clone the Repository (if applicable): git clone https://github.com/Liridon3/Song_Recommendation_Web_App.git
2. Set Up a Virtual Environment (Recommended)
- Navigate to the project directory.
- Create a virtual environment: 
  ```
  python -m venv venv
  ```
- Activate the virtual environment:
  - Windows: 
    ```
    venv\Scripts\activate
    ```
  - Unix/MacOS: 
    ```
    source venv/bin/activate
    ```

3. Install Dependencies:
    ```
    pip install -r requirements.txt
    ```
   

### Running the Application

1. **Launch the Flask App**
    ```
    python application.py
    ```
   or
    ```
    flask run
    ```
    
- The app will be accessible at `http://localhost:5000/`.

2. **Using the Application**
- Open a web browser and navigate to `http://localhost:5000/`.
- Follow the instructions on the website to interact with the app.

## Application Features

- **Song Clustering**: Implements KMeans to cluster songs based on features.
- **Song Recommendations**: Provides recommendations using cosine similarity.
- **Interactive Visualizations**: Includes visualizations like the Davies-Bouldin Index and genre popularity charts.

## Project Structure

- `app/`: Contains the Flask application, routes, and logic.
- `templates/`: HTML templates for the web interface.
- `spotify_songs.csv`: Dataset file with Spotify song metadata.
- `requirements.txt`: Lists the Python package dependencies.



