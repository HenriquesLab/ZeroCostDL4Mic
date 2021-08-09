
# FROM buildpack-deps:bionic
# FROM nvidia/container-toolkit:1.6.0-ubuntu18.04
# FROM datmo/buildpack-deps:gpu
FROM floydhub/buildpack-deps:cuda10.0-cudnn7-1.0
# packaging dependencies
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         dh-make \
#         fakeroot \
#         build-essential \
#         devscripts \
#         lsb-release && \
#     rm -rf /var/lib/apt/lists/*

# # packaging
# ARG PKG_VERS
# ARG PKG_REV
# ARG RUNTIME_VERSION
# ARG DOCKER_VERSION

# ENV DEBFULLNAME "NVIDIA CORPORATION"
# ENV DEBEMAIL "cudatools@nvidia.com"
# ENV REVISION "$PKG_VERS-$PKG_REV"
# ENV DOCKER_VERSION $DOCKER_VERSION
# ENV RUNTIME_VERSION $RUNTIME_VERSION
# ENV SECTION ""

# # output directory
# ENV DIST_DIR=/tmp/nvidia-docker2-$PKG_VERS
# RUN mkdir -p $DIST_DIR /dist

# # nvidia-docker 2.0
# COPY .binder/nvidia-docker $DIST_DIR/nvidia-docker
# COPY .binder/daemon.json $DIST_DIR/daemon.json

# WORKDIR $DIST_DIR
# COPY .binder/debian ./debian

# # RUN sed -i "s;@VERSION@;${REVISION};" debian/changelog && \
# #     sed -i "s;@VERSION@;${PKG_VERS};" $DIST_DIR/nvidia-docker && \
# #     if [ "$REVISION" != "$(dpkg-parsechangelog --show-field=Version)" ]; then echo "$(dpkg-parsechangelog --show-field=Version)" && exit 1; fi

# # CMD export DISTRIB="$(lsb_release -cs)" && \
# #     debuild --preserve-env --dpkg-buildpackage-hook='sh debian/prepare' -i -us -uc -b && \
# #     mv /tmp/*.deb /dist

# RUN set -eux; \
# 	apt-get update; \
# 	apt-get install -y --no-install-recommends \
# 		ca-certificates \
# 		curl \
# 		netbase \
# 		wget \
# # https://bugs.debian.org/929417
# 		tzdata \
# 	; \
# 	rm -rf /var/lib/apt/lists/*

# RUN set -ex; \
# 	if ! command -v gpg > /dev/null; then \
# 		apt-get update; \
# 		apt-get install -y --no-install-recommends \
# 			gnupg \
# 			dirmngr \
# 		; \
# 		rm -rf /var/lib/apt/lists/*; \
# 	fi

# # procps is very common in build systems, and is a reasonably small package
# RUN apt-get update && apt-get install -y --no-install-recommends \
# 		bzr \
# 		git \
# 		mercurial \
# 		openssh-client \
# 		subversion \
# 		\
# 		procps \
# 	&& rm -rf /var/lib/apt/lists/*


# RUN set -ex; \
# 	apt-get update; \
# 	apt-get install -y --no-install-recommends \
# 		autoconf \
# 		automake \
# 		bzip2 \
# 		dpkg-dev \
# 		file \
# 		g++ \
# 		gcc \
# 		imagemagick \
# 		libbz2-dev \
# 		libc6-dev \
# 		libcurl4-openssl-dev \
# 		libdb-dev \
# 		libevent-dev \
# 		libffi-dev \
# 		libgdbm-dev \
# 		libglib2.0-dev \
# 		libgmp-dev \
# 		libjpeg-dev \
# 		libkrb5-dev \
# 		liblzma-dev \
# 		libmagickcore-dev \
# 		libmagickwand-dev \
# 		libmaxminddb-dev \
# 		libncurses5-dev \
# 		libncursesw5-dev \
# 		libpng-dev \
# 		libpq-dev \
# 		libreadline-dev \
# 		libsqlite3-dev \
# 		libssl-dev \
# 		libtool \
# 		libwebp-dev \
# 		libxml2-dev \
# 		libxslt-dev \
# 		libyaml-dev \
# 		make \
# 		patch \
# 		unzip \
# 		xz-utils \
# 		zlib1g-dev \
# 		\
# # https://lists.debian.org/debian-devel-announce/2016/09/msg00000.html
# 		$( \
# # if we use just "apt-cache show" here, it returns zero because "Can't select versions from package 'libmysqlclient-dev' as it is purely virtual", hence the pipe to grep
# 			if apt-cache show 'default-libmysqlclient-dev' 2>/dev/null | grep -q '^Version:'; then \
# 				echo 'default-libmysqlclient-dev'; \
# 			else \
# 				echo 'libmysqlclient-dev'; \
# 			fi \
# 		) \
# 	; \
# 	rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update &&     apt-get -qq install --yes --no-install-recommends locales > /dev/null &&     apt-get -qq purge &&     apt-get -qq clean &&     rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen &&     locale-gen

ENV LC_ALL en_US.UTF-8

ENV LANG en_US.UTF-8

ENV LANGUAGE en_US.UTF-8

ENV SHELL /bin/bash

ARG NB_USER

ARG NB_UID

ENV USER ${NB_USER}

ENV HOME /home/${NB_USER}

RUN groupadd         --gid ${NB_UID}         ${NB_USER} &&     useradd         --comment "Default user"         --create-home         --gid ${NB_UID}         --no-log-init         --shell /bin/bash         --uid ${NB_UID}         ${NB_USER}

RUN wget --quiet -O - https://deb.nodesource.com/gpgkey/nodesource.gpg.key |  apt-key add - &&     DISTRO="bionic" &&     echo "deb https://deb.nodesource.com/node_14.x $DISTRO main" >> /etc/apt/sources.list.d/nodesource.list &&     echo "deb-src https://deb.nodesource.com/node_14.x $DISTRO main" >> /etc/apt/sources.list.d/nodesource.list

RUN apt-get -qq update &&     apt-get -qq install --yes --no-install-recommends        less        nodejs        unzip        > /dev/null &&     apt-get -qq purge &&     apt-get -qq clean &&     rm -rf /var/lib/apt/lists/*


EXPOSE 8888

ENV APP_BASE /srv

ENV NPM_DIR ${APP_BASE}/npm

ENV NPM_CONFIG_GLOBALCONFIG ${NPM_DIR}/npmrc

ENV CONDA_DIR ${APP_BASE}/conda

ENV NB_PYTHON_PREFIX ${CONDA_DIR}/envs/notebook

ENV KERNEL_PYTHON_PREFIX ${NB_PYTHON_PREFIX}

ENV PATH ${NB_PYTHON_PREFIX}/bin:${CONDA_DIR}/bin:${NPM_DIR}/bin:${PATH}

ENV NB_ENVIRONMENT_FILE="/tmp/environment.py-3.7.lock"

COPY --chown=${NB_USER}:${NB_USER} .binder/environment.py-3.7.lock ${NB_ENVIRONMENT_FILE}

COPY --chown=${NB_USER}:${NB_USER} .binder/activate_conda.sh /etc/profile.d/activate-conda.sh

RUN chmod +x /etc/profile.d/activate-conda.sh

COPY --chown=${NB_USER}:${NB_USER} .binder/environment.yml /tmp/environment.yml

COPY --chown=${NB_USER}:${NB_USER} .binder/install_miniforge.sh /tmp/install-miniforge.bash

RUN chmod +x /tmp/install-miniforge.bash

RUN mkdir -p ${NPM_DIR} && chown -R ${NB_USER}:${NB_USER} ${NPM_DIR}

USER ${NB_USER}

RUN npm config --global set prefix ${NPM_DIR}

USER root

RUN TIMEFORMAT='time: %3R' bash -c 'time /tmp/install-miniforge.bash' && rm /tmp/install-miniforge.bash ${NB_ENVIRONMENT_FILE}

ARG REPO_DIR=${HOME}

ENV REPO_DIR ${REPO_DIR}


WORKDIR ${REPO_DIR}

RUN chown ${NB_USER}:${NB_USER} ${REPO_DIR}


ENV PATH ${HOME}/.local/bin:${REPO_DIR}/.local/bin:${PATH}


ENV CONDA_DEFAULT_ENV ${KERNEL_PYTHON_PREFIX}


COPY --chown=${USER}:${USER} .binder/environment.yml ${REPO_DIR}/.binder/environment.yml

COPY --chown=${USER}:${USER} . ${REPO_DIR}

# RUN chmod +x "${REPO_DIR}/.binder/postBuild" && "${REPO_DIR}/.binder/postBuild"

USER ${NB_USER}


RUN TIMEFORMAT='time: %3R' bash -c 'time mamba env update -p ${NB_PYTHON_PREFIX} -f ".binder/environment.yml" && time mamba clean --all -f -y && mamba list -p ${NB_PYTHON_PREFIX} '


LABEL repo2docker.ref="None"

LABEL repo2docker.repo="local"

LABEL repo2docker.version="2021.01.0"


# COPY --chown=${USER}:${USER} . {REPO_DIR}

USER ${NB_USER}

# RUN chmod +x .binder/postBuild
 
RUN chmod +x "${REPO_DIR}/.binder/start"
ENV R2D_ENTRYPOINT "${REPO_DIR}/.binder/start"
# COPY /repo2docker-entrypoint /usr/local/bin/repo2docker-entrypoint
# ENTRYPOINT ["/usr/local/bin/repo2docker-entrypoint"]
CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]