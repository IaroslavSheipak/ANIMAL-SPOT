import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import AudioTimeline from "./AudioTimeline";
import "./styles.css";

const ConnectedAudioTimeline = withStreamlitConnection(AudioTimeline);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ConnectedAudioTimeline />
  </React.StrictMode>
);
