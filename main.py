from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse
from rag import run_rag

app = FastAPI()

# Serve HTML directly
@app.get("/")
def home():
    return FileResponse("templates/index.html")


@app.post("/")
async def query_ques_answer(
    query: str = Form(...),
    file: UploadFile = File(None)
):
    file_path = None

    if file:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

    answer = run_rag(query, file_path)

    return {"answer": answer}