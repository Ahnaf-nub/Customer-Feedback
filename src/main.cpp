#define EIDSP_QUANTIZE_FILTERBANK   0
#include "../lib/Speech_Emotion_inferencing/src/Speech_Emotion_inferencing.h"
#include "unihiker_k10.h"
#include <TFT_eSPI.h>
#include <driver/i2s.h>
#include <driver/gpio.h>
#include <WiFi.h>
#include <WebServer.h>

// Unihiker K10: ES7210 codec (4-channel TDM microphone array)

#define K10_SAMPLE_RATE         16000U  
#define K10_SAMPLE_BITS         16
#define M_SIZE 1.33 // Scale factor for display elements

#define WIFI_SSID "Mahir"

#define WIFI_PASSWORD "Ahnaf2007"

// K10 Board and Display
UNIHIKER_K10 k10;
TFT_eSPI tft = TFT_eSPI();
uint8_t screen_dir = 2;

// Display colors
#define TFT_GREY 0x5AEB
#define TFT_LIGHTGREEN 0x9772
#define INFO_PANEL_HEIGHT M_SIZE*40

// K10 Display Configuration
#define K10_DISPLAY_WIDTH       240
#define K10_DISPLAY_HEIGHT      320

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false;
static bool record_status = true;

static uint32_t emotion_counts[EI_CLASSIFIER_LABEL_COUNT] = {0};

#define K10_I2S_PORT I2S_NUM_0

static WebServer webServer(80);
static bool wifi_connected = false;
static String webserver_ip = "Connecting...";
static String labels_json;
static float web_probabilities[EI_CLASSIFIER_LABEL_COUNT] = {0};
static String web_last_timestamp_str = "waiting...";
static String web_top_label = "-";
static uint32_t inference_counter = 0;

static void pump_web_server();
static void setup_web_server();
static void handle_root_request();
static void handle_data_request();
static bool connect_to_wifi();
static String build_dashboard_page();
static String build_labels_json();

static esp_err_t init_i2s_mic_k10(void) {
    ei_printf("K10 I2S microphone already initialized by k10.begin()\n");
    ei_printf("Sample Rate: %d Hz, Bits: %d\n", K10_SAMPLE_RATE, K10_SAMPLE_BITS);
    return ESP_OK;
}

static void audio_inference_callback(uint32_t n_bytes) {
    for(int i = 0; i < n_bytes >> 1; i++) {
        inference.buffer[inference.buf_count++] = sampleBuffer[i];

        if(inference.buf_count >= inference.n_samples) {
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {
    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    static bool first_read = true;
    static int error_count = 0;
    
    while (record_status) {
        // Read from I2S microphone using legacy API (i2s_read)
        // Data comes in stereo format (L, R, L, R...)
        size_t bytes_read = 0;
        esp_err_t ret = i2s_read(K10_I2S_PORT, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, portMAX_DELAY);
        
        if (ret != ESP_OK) {
            error_count++;
            if (error_count <= 3) {  // Only print first 3 errors to avoid spam
                ei_printf("I2S read error: %d (error #%d)\n", ret, error_count);
            }
            continue;
        }
        
        if (bytes_read <= 0) {
            error_count++;
            if (error_count <= 3) {
                ei_printf("No bytes read from I2S\n");
            }
            continue;
        }
        
        if (first_read) {
            ei_printf("I2S reading successfully! Got %d bytes\n", bytes_read);
            first_read = false;
            error_count = 0;  // Reset error counter on successful read
        }
        
        // Use official UNIHIKER I2S mic stream at 16 kHz.
        // Stereo format: L, R, L, R, ... each 16-bit (2 bytes). Use left channel only.
        int16_t* stereo = (int16_t*)sampleBuffer;
        int16_t* mono = (int16_t*)sampleBuffer;
        int sample_count = bytes_read / 4;  // 2 channels * 2 bytes per sample
        
        for (int x = 0; x < sample_count; x++) {
            mono[x] = stereo[x * 2]; // left channel only
        }

        if (record_status) {
            audio_inference_callback(sample_count * 2);
        } else {
            break;
        }
    }
    vTaskDelete(NULL);
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    ei_sleep(100);
    record_status = true;

    xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, NULL);

    return true;
}

static bool microphone_inference_record(void) {
    while (inference.buf_ready == 0) {
        pump_web_server();
        delay(10);
    }

    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void) {
    record_status = false;
    ei_free(inference.buffer);
}

static const char index_page_template[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-900 text-slate-100">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>UNIHIKER K10 Emotion Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-b from-slate-900 via-slate-950 to-black">
    <div class="max-w-4xl mx-auto p-6 space-y-6">
        <header class="bg-slate-900/70 border border-slate-800 rounded-3xl p-6 backdrop-blur">
            <p class="text-sm uppercase tracking-[0.3em] text-emerald-400">Unihiker K10</p>
            <h1 class="text-3xl font-semibold text-white mt-2">Customer Feedback Radar</h1>
            <p class="text-slate-300 mt-4 text-sm">Webserver IP: <span class="font-mono text-emerald-300" id="ip">%DEVICE_IP%</span></p>
            <p class="text-slate-400 text-xs mt-1">Last inference: <span id="updated">waiting...</span></p>
        </header>
        <section class="bg-slate-900/70 border border-slate-800 rounded-3xl p-6 backdrop-blur">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-xl font-semibold">Live Emotion Confidence</h2>
                <span class="text-sm text-slate-400" id="inference-count">0 inferences</span>
            </div>
            <div id="bars" class="space-y-4"></div>
        </section>
        <section class="bg-slate-900/70 border border-slate-800 rounded-3xl p-6 backdrop-blur">
            <h2 class="text-xl font-semibold mb-3">Detection counters</h2>
            <div id="counters" class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm"></div>
        </section>
    </div>
    <script>
        const labels = %LABELS_JSON%;
        const gradients = ['from-rose-500 to-red-500','from-orange-400 to-amber-500','from-yellow-300 to-amber-300','from-emerald-400 to-lime-400','from-sky-400 to-cyan-400','from-indigo-400 to-purple-500'];
        function renderBars(data) {
            const barsEl = document.getElementById('bars');
            barsEl.innerHTML = labels.map((label, idx) => {
                const pct = Math.round((data.probabilities[idx] || 0) * 100);
                const gradient = gradients[idx % gradients.length];
                return `
                    <div class="space-y-2">
                        <div class="flex justify-between text-sm">
                            <span class="font-medium">${label}</span>
                            <span class="font-mono text-slate-300">${pct}%</span>
                        </div>
                        <div class="h-4 bg-slate-800/80 rounded-full overflow-hidden border border-slate-800">
                            <div class="h-full bg-gradient-to-r ${gradient} transition-all duration-500" style="width:${pct}%"></div>
                        </div>
                    </div>`;
            }).join('');
        }
        function renderCounters(data) {
            const countersEl = document.getElementById('counters');
            countersEl.innerHTML = labels.map((label, idx) => {
                const count = data.counts[idx] || 0;
                return `
                    <div class="bg-slate-950/80 border border-slate-800 rounded-2xl px-4 py-3 flex flex-col gap-1">
                        <span class="text-xs uppercase tracking-wide text-slate-400">${label}</span>
                        <span class="text-2xl font-semibold text-white">${count}</span>
                    </div>`;
            }).join('');
        }
        async function refreshDashboard() {
            try {
                const response = await fetch('/data');
                if (!response.ok) throw new Error('Network response was not ok');
                const data = await response.json();
                renderBars(data);
                renderCounters(data);
                document.getElementById('ip').textContent = data.ip;
                document.getElementById('updated').textContent = data.humanTimestamp || 'waiting...';
                document.getElementById('inference-count').textContent = `${data.inferenceCount || 0} inferences`;
            } catch (error) {
                console.error(error);
            }
        }
        refreshDashboard();
        setInterval(refreshDashboard, 1500);
    </script>
</body>
</html>
)rawliteral";

static String build_labels_json() {
        String json = "[";
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                if (ix > 0) json += ",";
                json += "\"";
                json += ei_classifier_inferencing_categories[ix];
                json += "\"";
        }
        json += "]";
        return json;
}

static String build_dashboard_page() {
        String page = FPSTR(index_page_template);
        page.replace("%DEVICE_IP%", webserver_ip);
        page.replace("%LABELS_JSON%", labels_json);
        return page;
}

static void handle_root_request() {
        webServer.send(200, "text/html", build_dashboard_page());
}

static void handle_data_request() {
        String json = "{";
        json += "\"ip\":\"" + webserver_ip + "\",";
        json += "\"humanTimestamp\":\"" + web_last_timestamp_str + "\",";
        json += "\"inferenceCount\":" + String(inference_counter) + ",";
        json += "\"topLabel\":\"" + web_top_label + "\",";
        json += "\"counts\":[";
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                if (ix > 0) json += ",";
                json += String(emotion_counts[ix]);
        }
        json += "],\"probabilities\":[";
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                if (ix > 0) json += ",";
                json += String(web_probabilities[ix], 5);
        }
        json += "],\"labels\":" + labels_json + "}";
        webServer.sendHeader("Cache-Control", "no-store");
        webServer.send(200, "application/json", json);
}

static void setup_web_server() {
        webServer.on("/", handle_root_request);
        webServer.on("/data", handle_data_request);
        webServer.onNotFound([](){ webServer.send(404, "text/plain", "Not found"); });
        webServer.begin();
        ei_printf("HTTP server started on http://%s\n", webserver_ip.c_str());
}

static bool connect_to_wifi() {
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        ei_printf("Connecting to WiFi SSID: %s\n", WIFI_SSID);
        unsigned long start = millis();
        const unsigned long timeout_ms = 20000;
        while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeout_ms) {
                delay(250);
                ei_printf(".");
        }
        ei_printf("\n");
        if (WiFi.status() == WL_CONNECTED) {
                wifi_connected = true;
                webserver_ip = WiFi.localIP().toString();
                ei_printf("WiFi connected, IP: %s\n", webserver_ip.c_str());
                return true;
        }
        wifi_connected = false;
        webserver_ip = "WiFi failed";
        ei_printf("WiFi connection failed. Update WIFI_SSID/WIFI_PASSWORD.\n");
        return false;
}

static void pump_web_server() {
        if (wifi_connected) {
                webServer.handleClient();
        }
}

static String format_uptime_timestamp() {
    unsigned long seconds = millis() / 1000UL;
    unsigned long minutes = seconds / 60UL;
    unsigned long hours = minutes / 60UL;
    seconds %= 60UL;
    minutes %= 60UL;
    char buffer[24];
    snprintf(buffer, sizeof(buffer), "T+%02luh%02lum%02lus", hours, minutes, seconds);
    return String(buffer);
}

// Display bar chart
static void display_emotion_chart(const ei_impulse_result_t* result) {
    tft.fillScreen(TFT_BLACK);
    
    // Title
    tft.setTextColor(TFT_CYAN, TFT_BLACK);
    tft.setTextFont(4);
    tft.drawString("Customer Feedback", M_SIZE*30, M_SIZE*10);

    // WiFi/IP info on top
    tft.setTextColor(TFT_LIGHTGREEN, TFT_BLACK);
    tft.setTextFont(2);
    String ipLine = "IP: " + webserver_ip;
    tft.drawString(ipLine.c_str(), M_SIZE*70, M_SIZE*35);
    
    // Timing info
    tft.setTextColor(TFT_GREY, TFT_BLACK);
    tft.setTextFont(2);
    
    // Draw emotion bars with detection counts
    int y_start = M_SIZE*60;
    int bar_height = M_SIZE*25;
    int bar_spacing = M_SIZE*5;
    uint16_t colors[] = {
        TFT_RED,        // RED
        TFT_ORANGE,     // ORANGE  
        TFT_YELLOW,     // YELLOW
        TFT_LIGHTGREEN  // GREEN
    };
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT && ix < 4; ix++) {
        const char* label = result->classification[ix].label;
        float value = result->classification[ix].value;
        
        // Draw label with detection count
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.setTextFont(2);
        char label_str[64];
        snprintf(label_str, sizeof(label_str), "%s: %u", label, emotion_counts[ix]);
        tft.drawString(label_str, M_SIZE*15, y_start - M_SIZE + (ix * (bar_height + bar_spacing)));
        
        // Draw bar background
        tft.drawRect(M_SIZE*80, y_start + (ix * (bar_height + bar_spacing)), M_SIZE*120, M_SIZE*20, TFT_GREY);
        
        // Draw filled bar (visual indicator of current confidence)
        int16_t bar_width = (int16_t)(M_SIZE*118 * value);
        if (bar_width > 0) {
            tft.fillRect(M_SIZE*81, y_start + 1 + (ix * (bar_height + bar_spacing)), bar_width, M_SIZE*18, colors[ix]);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    ei_printf("\n\n=== Unihiker K10 Speech Emotion Recognition ===\n");

    // Initialize K10 board (microphone, display, etc.)
    k10.begin();
    k10.initScreen(screen_dir);
    
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextFont(4);
    tft.drawString("Initializing K10...", M_SIZE*20, M_SIZE*100);

    // K10 microphone initialization is handled by k10.begin()
    ei_printf("K10 board initialized (microphone managed by k10.begin())\n");

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / K10_SAMPLE_RATE);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("\tClass %d: %s\n", ix, ei_classifier_inferencing_categories[ix]);
    }

    labels_json = build_labels_json();
    if (connect_to_wifi()) {
        setup_web_server();
    }
    
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d)\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        tft.fillScreen(TFT_BLACK);
        tft.setTextColor(TFT_RED, TFT_BLACK);
        tft.drawString("ERROR: Buffer Alloc", M_SIZE*20, M_SIZE*100);
        return;
    }

    ei_printf("Recording started...\n");
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.setTextFont(4);
    tft.drawString("Ready - Listening", M_SIZE*20, M_SIZE*150);
    ei_sleep(1000);
}

void loop() {
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    ei_printf("\n=== Predictions ===\n");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)\n",
              result.timing.dsp, result.timing.classification, result.timing.anomaly);

    // Print to serial
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: ", result.classification[ix].label);
        ei_printf_float(result.classification[ix].value);
        ei_printf("\n");
    }

    // Update shared web dashboard state
    inference_counter++;
    size_t top_idx = 0;
    float top_value = 0.0f;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        float value = result.classification[ix].value;
        web_probabilities[ix] = value;
        if (value > top_value) {
            top_value = value;
            top_idx = ix;
        }
    }
    if (top_value > 0.5f) {
        emotion_counts[top_idx]++;
    }
    web_top_label = String(ei_classifier_inferencing_categories[top_idx]);
    web_last_timestamp_str = format_uptime_timestamp();

    // Display bar chart on K10 screen
    display_emotion_chart(&result);
    
    // Small delay before next inference
    pump_web_server();
    delay(500);
    pump_web_server();
}