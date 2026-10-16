import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export type Params = Record<string, string | number | boolean | null | undefined>;

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  readonly base = environment.apiUrl;

  private params(params?: Params): HttpParams {
    let p = new HttpParams();
    for (const [k, v] of Object.entries(params ?? {})) {
      if (v !== null && v !== undefined && v !== '') p = p.set(k, String(v));
    }
    return p;
  }

  get<T>(path: string, params?: Params): Observable<T> {
    return this.http.get<T>(`${this.base}/${path}`, { params: this.params(params) });
  }
  post<T>(path: string, body: unknown = {}, headers?: Record<string, string>): Observable<T> {
    return this.http.post<T>(`${this.base}/${path}`, body, { headers });
  }
  patch<T>(path: string, body: unknown): Observable<T> {
    return this.http.patch<T>(`${this.base}/${path}`, body);
  }
  put<T>(path: string, body: unknown): Observable<T> {
    return this.http.put<T>(`${this.base}/${path}`, body);
  }
  delete<T>(path: string, body?: unknown): Observable<T> {
    return this.http.delete<T>(`${this.base}/${path}`, { body });
  }
  blob(path: string, params?: Params): Observable<Blob> {
    return this.http.get(`${this.base}/${path}`, { params: this.params(params), responseType: 'blob' });
  }

  /** Download a protected file (auth header required, so a plain link will not work). */
  download(path: string, filename: string, params?: Params): void {
    this.blob(path, params).subscribe(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  }
}
