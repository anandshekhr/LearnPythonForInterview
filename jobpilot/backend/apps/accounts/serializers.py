from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "email", "full_name", "date_joined"]
        read_only_fields = ["id", "email", "date_joined"]


class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    full_name = serializers.CharField(max_length=200, required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, min_length=10, style={"input_type": "password"})

    def validate_email(self, value):
        value = value.strip().lower()
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("An account with this email already exists.")
        return value

    def validate(self, attrs):
        validate_password(attrs["password"], user=User(email=attrs["email"], full_name=attrs.get("full_name", "")))
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["email"],
            email=validated_data["email"],
            full_name=validated_data.get("full_name", ""),
            password=validated_data["password"],
        )


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField()


class DeleteAccountSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True)
