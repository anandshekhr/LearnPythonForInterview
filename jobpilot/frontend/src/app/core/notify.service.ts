import { HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({ providedIn: 'root' })
export class NotifyService {
  private snack = inject(MatSnackBar);

  success(message: string): void {
    this.snack.open(message, 'OK', { duration: 3500 });
  }

  error(err: unknown, fallback = 'Something went wrong.'): void {
    this.snack.open(errorMessage(err, fallback), 'Dismiss', { duration: 7000, panelClass: 'snack-error' });
  }
}

/** Turn API error payloads ({error:{message}}, DRF field errors, detail) into readable text. */
export function errorMessage(err: unknown, fallback = 'Something went wrong.'): string {
  if (!(err instanceof HttpErrorResponse)) return fallback;
  if (err.status === 0) return 'Cannot reach the server. Check your connection.';
  const body = err.error;
  if (body?.error?.message) {
    const problems: string[] = body.error.details?.problems ?? [];
    return problems.length ? `${body.error.message} ${problems.join(' ')}` : body.error.message;
  }
  if (body?.detail) return body.detail;
  if (body && typeof body === 'object') {
    const parts = Object.entries(body).filter(([k]) => k !== 'request_id')
      .map(([k, v]) => `${k === 'non_field_errors' ? '' : k + ': '}${Array.isArray(v) ? v.join(' ') : v}`);
    if (parts.length) return parts.join(' · ');
  }
  return fallback;
}
