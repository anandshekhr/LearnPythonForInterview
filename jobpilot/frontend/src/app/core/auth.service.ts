import { HttpClient } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { Observable, finalize, map, shareReplay, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { User } from './models';

const REFRESH_KEY = 'jobpilot.refresh';

interface TokenPair { access: string; refresh: string; }

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private router = inject(Router);
  /** Access token lives in memory only; the refresh token is kept in localStorage to survive reloads. */
  private access = signal<string | null>(null);
  readonly user = signal<User | null>(null);
  readonly isLoggedIn = computed(() => !!this.access() || !!this.refreshToken());
  private refreshing$: Observable<string> | null = null;

  get accessToken(): string | null { return this.access(); }
  refreshToken(): string | null {
    try { return localStorage.getItem(REFRESH_KEY); } catch { return null; }
  }

  login(email: string, password: string): Observable<void> {
    return this.http.post<TokenPair>(`${environment.apiUrl}/auth/login/`, { email, password }).pipe(
      tap(t => this.store(t)), map(() => void 0));
  }

  register(email: string, password: string, full_name: string): Observable<void> {
    return this.http.post<TokenPair & { user: User }>(`${environment.apiUrl}/auth/register/`, { email, password, full_name }).pipe(
      tap(t => { this.store(t); this.user.set(t.user); }), map(() => void 0));
  }

  loadUser(): Observable<User> {
    return this.http.get<User>(`${environment.apiUrl}/auth/me/`).pipe(tap(u => this.user.set(u)));
  }

  refresh(): Observable<string> {
    if (!this.refreshing$) {
      this.refreshing$ = this.http.post<TokenPair>(`${environment.apiUrl}/auth/refresh/`, { refresh: this.refreshToken() }).pipe(
        tap(t => this.store(t)), map(t => t.access),
        finalize(() => (this.refreshing$ = null)), shareReplay(1));
    }
    return this.refreshing$;
  }

  logout(): void {
    const refresh = this.refreshToken();
    if (refresh && this.access()) {
      this.http.post(`${environment.apiUrl}/auth/logout/`, { refresh }).subscribe({ error: () => undefined });
    }
    this.clear();
    this.router.navigate(['/login']);
  }

  clear(): void {
    this.access.set(null);
    this.user.set(null);
    try { localStorage.removeItem(REFRESH_KEY); } catch { /* storage unavailable */ }
  }

  private store(t: TokenPair): void {
    this.access.set(t.access);
    try { if (t.refresh) localStorage.setItem(REFRESH_KEY, t.refresh); } catch { /* storage unavailable */ }
  }
}
