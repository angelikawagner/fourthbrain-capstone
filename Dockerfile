FROM python:3.9
WORKDIR /code
COPY ./fourthbrain_capstone_api/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./fourthbrain_capstone_api/app /code/app
# calling the main file app.py, so it's app.app instead of app.main
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]