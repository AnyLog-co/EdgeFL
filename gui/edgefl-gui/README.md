# EDGEFL GUI Demo

A modern React-based graphical user interface for the EDGEFL (Edge Federated Learning) demo. This application provides an intuitive step-by-step workflow for initializing, training, and running inference on federated learning models.

## Features

### 🔧 **Step-by-Step Workflow**
- **Initialize**: Configure node URLs and index name
- **Start Training**: Set training parameters and begin federated learning
- **Inference**: Run inference with various input data types

### 📊 **Multiple Input Types**
- **JSON Array**: Direct 28x28 array input with validation
- **JPG Images**: Image file upload (coming soon)
- **WAV Audio**: Audio file upload (coming soon)
- **Draw Canvas**: Interactive 28x28 grid for drawing
- **Text**: Shakespeare prompt continued by a training node

### 🎨 **Modern UI/UX**
- Clean blue and white design theme
- Responsive layout for desktop and mobile
- Real-time form validation and error handling
- Step-by-step navigation with progress indicators

### 🔄 **Smart Data Management**
- Shared index value across all steps
- Real-time synchronization between pages
- Automatic API endpoint construction
- Comprehensive error handling

## Quick Start

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd edgefl-gui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Start with Docker
1. Go to the GUI directory where the `package.json` file is located.
```bash
cd EdgeFL/gui/edgefl-gui
```
2. Build the docker image
```bash
docker build -t edgefl-gui:latest .
```
3. Start the docker container
```bash
docker run -p 3000:3000 edgefl-gui:latest
```
4. Go to http://127.0.0.1:3000/init

## Usage Guide

### Step 1: Initialize EDGEFL
1. **Set Node URL**: Enter your EDGEFL server address (e.g., `localhost:8080`)
2. **Configure Index**: Set the index name for your training session
3. **Add Node URLs**: Specify the federated learning nodes
4. **Initialize**: Click to start the initialization process

### Step 2: Start Training
1. **Training Parameters**: Set total rounds and minimum parameters
2. **Index Verification**: Confirm the index name from initialization
3. **Start Training**: Begin the federated learning process

### Step 3: Inference
1. **Choose Input Type**: Select from JSON, JPG, WAV, Draw Canvas, or Text
2. **Provide Data**: 
   - **JSON**: Enter or upload a 28x28 array
   - **Draw Canvas**: Click and drag on the 28x28 grid
   - **Text**: Type a line and set the training-node URL. The node returns a continuation.
3. **Run Inference**: Process your data through the trained model

## API Endpoints

The application communicates with the following EDGEFL endpoints:

- **POST `/init`**: Initialize the federated learning setup
- **POST `/start-training`**: Begin the training process
- **POST `/infer`**: Run inference on input data

### Adding New Features

1. **New Input Types**: Extend `InputDataSelector.js`
2. **API Endpoints**: Add functions to `services/api.js`
3. **Pages**: Create new components in `pages/`
4. **Styling**: Update `styles/App.css`


---

**Note**: JPG and WAV file processing features are currently in development and will be available in future updates.