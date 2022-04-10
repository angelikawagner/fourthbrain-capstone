# 1. Library imports
from imp import reload
import uvicorn
from fastapi import FastAPI
from Model import MyLdaModel, InputDoc

# 2. Create app and model objects
app = FastAPI()
model = MyLdaModel()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_topics(a_doc: InputDoc):
    data = a_doc.dict()
    topics = model.predict_topics(data['input_doc'])
    return {
        'topics': str(topics),
        'num_topics': len(topics) 
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    # uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True)