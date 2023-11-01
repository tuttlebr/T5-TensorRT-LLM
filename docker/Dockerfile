ARG BASE_IMAGE_TAG

FROM ${BASE_IMAGE_TAG} as release

WORKDIR /tmp
COPY requirements.txt .
RUN pip3 install -r requirements.txt

FROM release as devel

RUN pip3 install jupyterlab theme-darcula

RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension \
    && echo '{ "theme": "Darcula" }' > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings \
    && jupyter labextension disable "@jupyterlab/apputils-extension:announcements"