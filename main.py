from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from PIL import Image
import google.generativeai as genai
import uvicorn

# Configure the API key directly in the script
API_KEY = 'AIzaSyB4XAeahNPcXlpA0UUFLYOZ1up2llYEoNg'
genai.configure(api_key=API_KEY)

# Generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Model name
MODEL_NAME = "gemini-1.5-pro-latest"

# Framework selection (e.g., Tailwind, Bootstrap, etc.)
framework = "Regular CSS use flex grid etc"  # Change this to "Bootstrap" or any other framework as needed

# Create the model
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat(history=[])

# Function to send a message to the model
def send_message_to_model(message, image_path):
    image_input = {
        'mime_type': 'image/jpeg',
        'data': pathlib.Path(image_path).read_bytes()
    }
    response = chat_session.send_message([message, image_input])
    return response.text

# FastAPI app
app = FastAPI(
    title="Gemini 1.5 Pro UI to Code",
    description="Convert UI images to HTML code with Gemini 1.5 Pro",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate_code")
async def generate_code(image: UploadFile = File(...)):
    try:
        # Load and display the image
        image = Image.open(image.file)

        # Convert image to RGB mode if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Save the uploaded image temporarily
        temp_image_path = pathlib.Path("temp_image.jpg")
        image.save(temp_image_path, format="JPEG")

        # Generate UI description
        prompt = "Jelaskan UI ini dengan detail yang akurat. Saat Anda merujuk ke elemen UI, letakkan namanya dan kotak pembatasnya dalam format: [nama objek (y_min, x_min, y_max, x_max)]. Jelaskan juga warna dari elemen tersebut."
        description = send_message_to_model(prompt, temp_image_path)

        # Refine the description
        refine_prompt = f"Bandingkan elemen UI yang dijelaskan dengan gambar yang diberikan dan identifikasi elemen atau ketidakakuratan yang hilang. Jelaskan juga warna dari elemen tersebut. Berikan deskripsi yang akurat dan disempurnakan dari elemen UI berdasarkan perbandingan ini. Berikut adalah deskripsi awal: {description}"
        refined_description = send_message_to_model(refine_prompt, temp_image_path)

        # Generate HTML
        html_prompt = f"Buat file HTML berdasarkan deskripsi UI berikut, menggunakan elemen UI yang dijelaskan dalam respons sebelumnya. Sertakan CSS {framework} dalam file HTML untuk menyusun elemen. Pastikan warna yang digunakan sama dengan UI asli. UI perlu responsif dan mobile-first, mencocokkan UI asli semaksimal mungkin. Jangan menyertakan penjelasan atau komentar apa pun. Hindari menggunakan ```html. dan ``` di akhir. HANYA kembalikan kode HTML dengan CSS inline. Berikut adalah deskripsi yang disempurnakan: {refined_description}"
        initial_html = send_message_to_model(html_prompt, temp_image_path)

        # Refine HTML
        refine_html_prompt = f"Validasi kode HTML berikut berdasarkan deskripsi UI dan gambar dan berikan versi yang disempurnakan dari kode HTML dengan CSS {framework} yang meningkatkan akurasi, responsivitas, dan kepatuhan terhadap desain asli. HANYA kembalikan kode HTML yang disempurnakan dengan CSS inline. Hindari menggunakan ```html. dan ``` di akhir. Berikut adalah HTML awal: {initial_html}"
        refined_html = send_message_to_model(refine_html_prompt, temp_image_path)

        return {
            "html_code": refined_html,
            "status": "success",
            "message": "Kode HTML berhasil dihasilkan!"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Terjadi kesalahan: {e}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
