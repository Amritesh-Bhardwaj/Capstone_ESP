# System Architecture

```mermaid
flowchart TD
    subgraph Hardware Layer
        AP[WiFi Access Point / Transmitter]
        ESP[ESP32 Board / Receiver]
        AP -- "WiFi (802.11, 20MHz)" --> ESP
    end

    subgraph Backend Layer (Python)
        Serial[Serial Reader]
        Pipeline[CSI Preprocessing Pipeline]
        LSTM[LSTM Inference Model]
        API[FastAPI / WebSocket Server]
        
        ESP -- "Raw CSI (Serial)" --> Serial
        Serial -- "Valid Frames" --> Pipeline
        Pipeline -- "(n, 64, 48) Tensors" --> LSTM
        Pipeline -- "Processed Data" --> API
        LSTM -- "Predicted Activity" --> API
    end

    subgraph Frontend Layer (Web GUI)
        UI[Web Dashboard]
        Viz[CSI Real-Time Heatmap]
        Status[Live Activity Indicator]
        
        API -- "WebSocket JSON Stream" --> UI
        UI --> Viz
        UI --> Status
    end

    classDef hardware fill:#e1f5fe,stroke:#0288d1;
    classDef backend fill:#e8f5e9,stroke:#388e3c;
    classDef frontend fill:#fff3e0,stroke:#f57c00;

    class AP,ESP hardware;
    class Serial,Pipeline,LSTM,API backend;
    class UI,Viz,Status frontend;
```
