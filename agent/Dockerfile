FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent_app.py /app/agent_app.py

EXPOSE 9001
CMD ["uvicorn", "agent_app:app", "--host", "0.0.0.0", "--port", "9001"]
