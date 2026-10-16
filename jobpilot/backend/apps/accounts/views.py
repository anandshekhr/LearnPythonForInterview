from django.contrib.auth import get_user_model
from drf_spectacular.utils import extend_schema
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from apps.core import audit
from apps.core.exceptions import DomainError

from .serializers import DeleteAccountSerializer, LogoutSerializer, RegisterSerializer, UserSerializer

User = get_user_model()


class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_scope = "auth"

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        audit.record("user.registered", user=user, obj=user, request=request)
        refresh = RefreshToken.for_user(user)
        return Response(
            {"user": UserSerializer(user).data, "access": str(refresh.access_token), "refresh": str(refresh)},
            status=status.HTTP_201_CREATED,
        )


class LoginView(TokenObtainPairView):
    throttle_scope = "auth"


class RefreshView(TokenRefreshView):
    throttle_scope = "auth"


class LogoutView(APIView):
    @extend_schema(request=LogoutSerializer, responses={205: None})
    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            RefreshToken(serializer.validated_data["refresh"]).blacklist()
        except TokenError:
            raise DomainError("Invalid refresh token.", code="invalid_token")
        audit.record("user.logout", request=request)
        return Response(status=status.HTTP_205_RESET_CONTENT)


class MeView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user

    @extend_schema(request=DeleteAccountSerializer, responses={204: None})
    def delete(self, request):
        """Permanently delete the account and all owned data (resumes, documents, applications)."""
        serializer = DeleteAccountSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        if not request.user.check_password(serializer.validated_data["password"]):
            raise DomainError("Password is incorrect.", code="invalid_password", status_code=403)
        from apps.profiles.services.storage import delete_user_files

        user = request.user
        audit.record("user.deleted", user=None, obj=user, request=request, email_domain=user.email.split("@")[-1])
        delete_user_files(user)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
