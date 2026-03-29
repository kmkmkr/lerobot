# uv sync --extra feetech --extra hilserl --extra async --extra pi --extra test
uv venv
source .venv/bin/activate
uv pip install -e ".[feetech,hilserl,async,pi,test]"