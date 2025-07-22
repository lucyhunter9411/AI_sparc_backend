from pydantic import BaseModel, EmailStr, SecretStr, model_validator

class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr

class RegisterRequest(BaseModel):
    email: EmailStr
    validation_email: EmailStr
    password: SecretStr

    @model_validator(mode="after")
    def emails_match(self):
        if self.email != self.validation_email:
            raise ValueError('Email and validation_email must match')
        return self