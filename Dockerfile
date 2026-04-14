FROM ghcr.io/prefix-dev/pixi:0.65.0 AS build

WORKDIR /app
COPY pixi.toml pixi.lock pyproject.toml ./
ENV CONDA_OVERRIDE_CUDA=12.0
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates > /dev/null && rm -rf /var/lib/apt/lists/*

# Build target env: docker (ykkcadparse) or vllm (glm-ocr)
ARG PIXI_ENV=docker
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked -e ${PIXI_ENV}
RUN pixi shell-hook -e ${PIXI_ENV} -s bash > /shell-hook.sh

FROM ubuntu:24.04 AS production

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ca-certificates > /dev/null && rm -rf /var/lib/apt/lists/*

ARG PIXI_ENV=docker
WORKDIR /app
COPY --from=build /app/.pixi/envs/${PIXI_ENV} /app/.pixi/envs/${PIXI_ENV}
COPY --from=build /shell-hook.sh /shell-hook.sh

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Only copy ykkcadparse source for the docker target
COPY src/ykkcadparse/ /app/ykkcadparse/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["bash", "-c", "source /shell-hook.sh && exec uvicorn ykkcadparse.server:app --host 0.0.0.0 --port 8000"]
