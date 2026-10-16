import { HttpErrorResponse, HttpInterceptorFn, HttpRequest } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, switchMap, throwError } from 'rxjs';
import { AuthService } from './auth.service';

const withToken = (req: HttpRequest<unknown>, token: string | null) =>
  token ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req;

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  if (req.url.includes('/auth/login') || req.url.includes('/auth/register') || req.url.includes('/auth/refresh')) {
    return next(req);
  }
  const send = (token: string | null) => next(withToken(req, token));
  const start = !auth.accessToken && auth.refreshToken() ? auth.refresh() : null;
  const first$ = start ? start.pipe(switchMap(t => send(t))) : send(auth.accessToken);
  return first$.pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401 && auth.refreshToken() && !start) {
        return auth.refresh().pipe(
          switchMap(t => send(t)),
          catchError(e => { auth.logout(); return throwError(() => e); }));
      }
      if (err.status === 401) auth.logout();
      return throwError(() => err);
    }));
};
