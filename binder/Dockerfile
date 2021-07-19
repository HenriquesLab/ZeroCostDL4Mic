# # FROM tensorflow/tensorflow:latest-gpu-jupyter

# # # --- Jupyter

# # # install the notebook package
# # RUN pip install --no-cache --upgrade pip && \
# #     pip install --no-cache notebook

# # ENV HOME /content
# # # RUN mkdir ${HOME}

# # # RUN chown ${NB_USER}:${NB_USER} -R ${HOME}

# # # create user with a home directory
# # ARG NB_USER
# # ARG NB_UID
# # ENV USER ${NB_USER}
# # # create user with a home directory
# # ARG NB_USER
# # ARG NB_UID
# # ENV USER ${NB_USER}
# # ENV HOME /home/${NB_USER}

# # RUN adduser --disabled-password \
# #     --gecos "Default user" \
# #     --uid ${NB_UID} \
# #     ${NB_USER}
# # WORKDIR ${HOME}
# # USER ${USER}

# # # RUN adduser --disabled-password \
# # #     --gecos "Default user" \
# # #     --uid ${NB_UID} \
# # #     --home ${HOME} \
# # #     ${NB_USER}
# # # # WORKDIR ${HOME}
# # # # USER ${USER}

# # # # # RUN useradd ${USER} --home ${HOME} --create-home
# # # # # RUN useradd -rm -d /content -s /bin/bash ${USER} --create-home
# # # # # RUN adduser --disabled-password --gecos '' ${USER} -S -D -h /content
# # # # WORKDIR ${HOME}
# # # # # # RUN chown ${USER}:${USER} -R ${HOME}
# # # # USER ${USER}

# # # # # # RUN conda install pip --yes

# # # # COPY . .

# # # # RUN pip install --no-cache-dir -r requirements.txt


# # # # # create user with a home directory

# FROM python:3.7-slim
# # install the notebook package
# RUN pip install --no-cache --upgrade pip && \
#     pip install --no-cache notebook

# # create user with a home directory
# ARG NB_USER
# ARG NB_UID
# ENV USER ${NB_UID}
# ENV HOME /home/{USER}

# RUN adduser --disabled-password \
#     --gecos "Default user" \
#     --uid ${NB_UID} \
#     --home ${HOME} \
#     ${NB_USER}

# WORKDIR ${HOME}
# USER ${NB_UID}

# COPY --chown=${NB_UID}:${NB_UID} . .

# RUN chown ${NB_UID}:${NB_UID} -R ${HOME}
# RUN chmod 777 ${HOME}

# RUN ls -al
# # ENV PATH="${PATH}:{HOME}/.local/bin"
# RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.7-slim
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook

RUN  apt-get update \
  && apt-get install -y wget rclone curl nodejs npm

RUN npm --version 


ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
ENV COLAB /content

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

# RUN mkdir ${COLAB}
RUN ln -s /home/${USER}/ /${COLAB}
# RUN ln -s /home/${USER} /${COLAB}

WORKDIR ${COLAB}
RUN cd ${COLAB}


COPY --chown=${USER}:${USER} . .
# RUN ls -al
RUN pip install --no-cache-dir -r requirements.txt

USER ${USER}

RUN jupyter labextension install @eddienko/jupyterlab-colabinspired-codecellbtn \
                             @eddienko/jupyterlab-colabinspired-theme-light \
                             imjoy-jupyter-extension \
                             @jupyterlab/toc @jupyterlab/git
# RUN jupyter labextension install imjoy-jupyter-extension  

