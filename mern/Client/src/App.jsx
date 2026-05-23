
import React from "react";
import {
  Navigate,
  Route,
  BrowserRouter as Router,
  Routes,
} from "react-router-dom";
import Analyze from "./Analyze";
import Blogs from "./Blogs";
import CureLinks from "./CureLinks"; // Import the new CureLinks component
import LandingPage from "./LandingPage";
import Login from "./Login";
import Register from "./Register";

// Use explicit boolean conversion for clarity
const isAuthenticated = () => Boolean(localStorage.getItem("token"));

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route
          path="/analyze"
          element={isAuthenticated() ? <Analyze /> : <Navigate to="/login" />}
        />
        <Route path="/blogs" element={<Blogs />} />
        {/* New CureLinks route */}
        <Route path="/cure-links/:disease" element={<CureLinks />} />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}

export default App;
