import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

type ErrorBoundaryState = {
  error: Error | null;
};

class AppErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("CUBIT Inspect render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <main
          style={{
            minHeight: "100vh",
            boxSizing: "border-box",
            display: "grid",
            placeItems: "center",
            padding: "32px",
            background: "#101820",
            color: "#e8edf1",
            fontFamily: "Inter, system-ui, sans-serif",
          }}
        >
          <section style={{ maxWidth: "620px", width: "100%" }}>
            <p style={{ color: "#e6a47e", fontSize: "11px", letterSpacing: "1.5px" }}>
              CUBIT INSPECT · RENDER ERROR
            </p>
            <h1 style={{ fontSize: "24px", marginBottom: "10px" }}>
              The inspection view encountered an error.
            </h1>
            <p style={{ color: "#aab5bf", lineHeight: 1.6 }}>
              The application caught a frontend rendering exception instead of showing a blank window.
              Share this message when reporting the issue:
            </p>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                overflowWrap: "anywhere",
                padding: "16px",
                border: "1px solid #35424e",
                borderRadius: "8px",
                background: "#17212b",
                color: "#f0b49a",
                fontSize: "12px",
              }}
            >
              {this.state.error.message}
            </pre>
            <button
              onClick={() => this.setState({ error: null })}
              style={{
                marginTop: "12px",
                padding: "10px 14px",
                border: 0,
                borderRadius: "6px",
                background: "#bd7957",
                color: "#fff",
                cursor: "pointer",
              }}
            >
              Try rendering again
            </button>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>,
);
