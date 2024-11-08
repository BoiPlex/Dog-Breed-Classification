import './App.css';

import logo from './dog-logo.svg';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h1>Welcome to the Dog Breed Classifier</h1>
        <p>
          Our dog breed classifier is currently under development.
        </p>
        <p>Stay tuned for updates!</p>
      </header>
    </div>
  );
}

export default App;