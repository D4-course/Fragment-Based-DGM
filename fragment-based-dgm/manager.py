from fastapi import FastAPI
import uvicorn
import os
from fastapi.responses import FileResponse, StreamingResponse
import io
import zipfile


BASE_API = "/fragment-based-dgm"
RUN_DIR = "./RUNS/2022-11-07@22_05_41-SpeedForce-PCBA"
SAMPLE_DIR = "./RUNS/2022-11-07@22_05_41-SpeedForce-PCBA/results/samples/2022-11-08@16_39_14_30.csv"
DATASET = ""

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/{preprocess}")
def preprocess_model(dataset: str):
    DATASET = dataset
    cmdline = "python3 manage.py preprocess --dataset " + DATASET
    os.system(cmdline)
    return  {"message": "Dataset: " + DATASET + "Preprocessed"}

@app.post("/{train}")
def train_model():
    cmdline = "python3 manage.py preprocess --dataset " + DATASET + "--use_gpu"
    os.system(cmdline)
    return  {"message": "Trained on: " + DATASET}

@app.post("/{sample}")
def sample_model():
    cmdline = "python3 manage.py sample --run " + RUN_DIR
    os.system(cmdline)
    with open(SAMPLE_DIR) as f:
        samples = f.read()
        return StreamingResponse(io.BytesIO(samples), media_type= "text/csv", headers={"Content-Disposition": "attachment;filename = 2022-11-08@16_39_14_30.csv"})

@app.post("/{postprocess}")
def postprocess_model():
    cmdline = "python3 manage.py postprocess --run " + RUN_DIR
    os.system(cmdline)
    return   {"message": "Postprocessing done, proceed to plotting"}

@app.post("/{plot}")
def plot_model():
    cmdline = "python3 manage.py plot --run " + RUN_DIR
    os.system(cmdline)
    plot1 = "counts_"+DATASET+".svg"
    plot2 = "props_"+DATASET+".svg"
    zip_subdir = "./"
    filenames = [plot1, plot2]

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode="w", compresion = zipfile.ZIP_DEFLATED) as tmp_zip:
        for fpath in filenames:
            # Calculate path for file in zip
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            temp_zip.write(fpath, zip_path)
        return StreamingResponse(iter([zip_io.getvalue()]), media_type="application/x-zip-compressed", headers = { "Content-Disposition": f"attachment; filename=images.zip"})

if __name__ == "__main__":
    uvicorn.run("manager:app", host="0.0.0.0", port = 8080)
