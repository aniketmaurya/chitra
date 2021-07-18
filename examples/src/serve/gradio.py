from chitra.serve.app import GradioApp

app = GradioApp("image-classification", model=lambda x: 12121)

app.run()
