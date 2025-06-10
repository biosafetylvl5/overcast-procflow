FROM geoips-python3.11

# Define build-time arguments with default values
ARG USER=geoips_user
ARG GROUP=${USER}
ARG PLUGIN=PLUGIN

# Set environment variables for use during build and runtime
ENV PLUGIN_NAME=${PLUGIN}
ENV PLUGIN_DIR="$GEOIPS_PACKAGES_DIR/$PLUGIN_NAME"
ENV PLUGIN_DATA_DIR="$GEOIPS_TESTDATA_DIR/$PLUGIN_NAME"

USER root

# Create the group and user if they don't exist, and set up directories
RUN mkdir -p "$PLUGIN_DIR" "$PLUGIN_DATA_DIR"; \
    chmod -R a+rw "$PLUGIN_DATA_DIR" "$PLUGIN_DIR"

# Switch to the new user
USER ${USER}

# Set the working directory
WORKDIR ${PLUGIN_DIR}

# Copy and install Python dependencies separately to leverage caching
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --user -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install the plugin in editable mode
RUN python3 -m pip install --user -e ./dependencies/overcast_preprocessing; \
    python3 -m pip install --user -e ./dependencies/octopy; \
    python3 -m pip install --user -e ./dependencies/geoips_clavrx; \
    python3 -m pip install --user -e .[doc,lint,test]; \
    create_plugin_registries;

USER root
RUN chmod -R a+rw "$PLUGIN_DATA_DIR" "$PLUGIN_DIR"
USER ${USER}

# Update PATH for the user's local bin directory
ENV PATH=$PATH:/home/${USER}/.local/bin


