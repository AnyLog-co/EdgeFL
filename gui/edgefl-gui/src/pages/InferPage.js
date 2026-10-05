import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useServer } from '../contexts/ServerContext';
import {runInference, validateInputArray, generateSampleArray, validateAndProcessImage, evaluateTestSet, inspectShakespearePrompt, SHAKESPEARE_SAMPLE} from '../services/api';
import InputDataSelector from '../components/InputDataSelector';

const InferPage = () => {
  const navigate = useNavigate();
  const { serverUrl, indexValue, setIndexValue } = useServer();
  const [inputData, setInputData] = useState('');
  const [inputType, setInputType] = useState('json');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [testEvalLoading, setTestEvalLoading] = useState(false);
  const [testEvalResponse, setTestEvalResponse] = useState(null);
  const [testEvalError, setTestEvalError] = useState(null);
  const [textNodeUrl, setTextNodeUrl] = useState('localhost:8081');

  const textCheck = inputType === 'text' ? inspectShakespearePrompt(inputData) : null;

  const resolveUrl = (value) => (
    value.startsWith('http://') || value.startsWith('https://') ? value : `http://${value}`
  );

  const generateSampleData = () => {
    const array = generateSampleArray();
    setInputData(JSON.stringify(array, null, 2));
  };

  const handleDataChange = (data, type) => {
    setInputData(data);
    setInputType(type);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);

    let inputArray;
    try {

      if (inputType === 'json') {
        inputArray = validateInputArray(inputData);
      } else if (inputType === 'png' || inputType === 'jpg' || inputType === 'wav') {
        // For file uploads, we'll need to process the file
        // For now, we'll show an error that this feature is coming soon

        try {
            const floatArray = await validateAndProcessImage(inputData);
            console.log('Float32Array:', floatArray);

            inputArray = Array.from(floatArray);
            console.log('Converted to regular array:', inputArray);

            // You can now send `inputArray` to your FastAPI backend
          } catch (error) {
            console.error('Error processing image:', error.message);
          }

          // console.log(inputArray)
        // throw new Error(`${inputType.toUpperCase()} file processing is coming soon!`);
      } else if (inputType === 'draw') {
        // For grid drawings, the data is already in the correct format
        inputArray = typeof inputData === 'string' ? JSON.parse(inputData) : inputData;
      } else if (inputType === 'text') {
        const encoded = inspectShakespearePrompt(inputData);
        if (encoded.ids.length === 0) {
          throw new Error('Enter a prompt that includes characters from the Shakespeare vocabulary.');
        }
        inputArray = encoded.ids;
      }

      console.log("FINAL ARRAY:", inputArray)
      const inferenceUrl = inputType === 'text' ? resolveUrl(textNodeUrl) : serverUrl;
      const data = await runInference(inferenceUrl, { input: inputArray, index: indexValue });
      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTestSetEvaluation = async () => {
    setTestEvalLoading(true);
    setTestEvalError(null);
    setTestEvalResponse(null);

    try {
      const evaluationUrl = inputType === 'text' ? resolveUrl(textNodeUrl) : serverUrl;
      const data = await evaluateTestSet(evaluationUrl, indexValue);
      setTestEvalResponse(data);
    } catch (err) {
      setTestEvalError(err.message);
    } finally {
      setTestEvalLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Step 3: Inference</h1>
        <p>Run inference on the trained model</p>
      </div>

      <form onSubmit={handleSubmit} className="form-container">
        <div className="info-box">
          <h3>Inference Configuration</h3>
          <p>Choose your input data type and provide the data for inference.</p>
        </div>

        <div className="form-group">
          <label htmlFor="index">Index Name:</label>
          <input
            type="text"
            id="index"
            value={indexValue}
            onChange={(e) => setIndexValue(e.target.value)}
            placeholder="test-index"
            required
          />
          <small>Index name to use for inference</small>
        </div>

        <InputDataSelector
          inputData={inputData}
          setInputData={setInputData}
          onDataChange={handleDataChange}
        />

        {inputType === 'text' && (
          <div className="form-group">
            <label htmlFor="textNodeUrl">Training node:</label>
            <input
              type="text"
              id="textNodeUrl"
              value={textNodeUrl}
              onChange={(e) => setTextNodeUrl(e.target.value)}
              placeholder="localhost:8081"
              required
            />
            <small>
              Live text prediction is sent to this training node, not the aggregator in the header.
              The node must already be initialized with the index above.
            </small>
          </div>
        )}

        {textCheck && textCheck.dropped.length > 0 && (
          <div className="info-box">
            <p>
              Ignored characters outside the vocabulary: {textCheck.dropped.join(' ')}
            </p>
          </div>
        )}

        <div className="button-group">
          {inputType === 'json' && (
            <button type="button" onClick={generateSampleData} className="btn-secondary">
              Generate Sample Data
            </button>
          )}
          {inputType === 'text' && (
            <button
              type="button"
              onClick={() => handleDataChange(SHAKESPEARE_SAMPLE, 'text')}
              className="btn-secondary"
            >
              Use Sample Line
            </button>
          )}
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? 'Running Inference...' : 'Run Inference'}
          </button>
        </div>

        <div className="button-group" style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '20px' }}>
          <button 
            type="button" 
            onClick={handleTestSetEvaluation} 
            disabled={testEvalLoading || !indexValue.trim()} 
            className="btn-primary"
            style={{ backgroundColor: '#28a745' }}
          >
            {testEvalLoading ? 'Evaluating Test Set...' : 'Evaluate Test Set'}
          </button>
          <small style={{ display: 'block', marginTop: '5px', color: '#666' }}>
            Run model evaluation against the test dataset for index: {indexValue || 'test-index'}
          </small>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <h3>Error:</h3>
          <p>{error}</p>
        </div>
      )}

      {response && (
        <div className="success-message">
          <h3>Inference Results:</h3>
          {response.prediction && typeof response.prediction === 'object' && response.prediction.continuation ? (
            <div className="text-prediction">
              <p className="text-prediction-line">
                <span>{response.prediction.prompt}</span>
                <span className="continuation">{response.prediction.continuation}</span>
              </p>
              {Array.isArray(response.prediction.top_k) && (
                <ul className="top-k-list">
                  {response.prediction.top_k.map((item) => (
                    <li key={`${item.char}-${item.probability}`}>
                      {item.char === '\n' ? '\\n' : item.char} ({item.probability})
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <pre>{JSON.stringify(response, null, 2)}</pre>
          )}
        </div>
      )}

      {testEvalError && (
        <div className="error-message">
          <h3>Test Set Evaluation Error:</h3>
          <p>{testEvalError}</p>
        </div>
      )}

      {testEvalResponse && (
        <div className="success-message">
          <h3>Test Set Evaluation Results:</h3>
          <pre>{JSON.stringify(testEvalResponse, null, 2)}</pre>
        </div>
      )}

      <div className="navigation-buttons">
        <button onClick={() => navigate('/start-training')} className="btn-secondary">
          ← Previous
        </button>
      </div>
    </div>
  );
};

export default InferPage;
