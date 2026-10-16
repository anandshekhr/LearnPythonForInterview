from rest_framework.permissions import SAFE_METHODS, BasePermission


class IsOwner(BasePermission):
    """Object-level check. Querysets are already filtered by owner; this is defence in depth."""

    owner_field = "user"

    def has_object_permission(self, request, view, obj):
        owner_path = getattr(view, "owner_field", self.owner_field)
        target = obj
        for part in owner_path.split("__"):
            target = getattr(target, part, None)
        return target is not None and getattr(target, "pk", target) == request.user.pk


class IsStaffOrReadOnly(BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and (request.method in SAFE_METHODS or request.user.is_staff))


class OwnedQuerysetMixin:
    """Restricts a ViewSet's queryset to the requesting user's rows and sets the owner on create."""

    owner_field = "user"

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return super().get_queryset().none()
        return super().get_queryset().filter(**{self.owner_field: self.request.user})

    def perform_create(self, serializer):
        if "__" in self.owner_field:
            serializer.save()
        else:
            serializer.save(**{self.owner_field: self.request.user})
