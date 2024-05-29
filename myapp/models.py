from django.db import models
from django.contrib.auth.hashers import make_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Adjust max_length to match the hash length
    
    def save(self, *args, **kwargs):
        # Hash the password before saving
        self.password = make_password(self.password)
        super().save(*args, **kwargs)
    
    def clean(self):
        # Validate email format
        try:
            validate_email(self.email)
        except ValidationError:
            raise ValidationError("Invalid email format.")

    def __str__(self):
        return self.username

