FROM public.ecr.aws/lambda/python:3.8

# Create function directory
WORKDIR /app

# Install the function's dependencies
# Copy file requirements.txt from your project folder and install
# the requirements in the app directory.

# CHITRA-FILE-COPY-PLACEHOLDER

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt

# Copy handler function (from the local app directory)
COPY  app.py  .

# Overwrite the command by providing a different command directly in the template.
CMD ["/app/app.handler"]
