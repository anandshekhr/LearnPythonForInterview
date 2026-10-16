import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { authInterceptor } from './auth.interceptor';
import { AuthService } from './auth.service';

describe('authInterceptor', () => {
  let http: HttpClient;
  let ctrl: HttpTestingController;
  let auth: AuthService;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({ providers: [provideRouter([]), provideHttpClient(withInterceptors([authInterceptor])), provideHttpClientTesting()] });
    http = TestBed.inject(HttpClient);
    ctrl = TestBed.inject(HttpTestingController);
    auth = TestBed.inject(AuthService);
  });
  afterEach(() => { ctrl.verify(); localStorage.clear(); });

  it('refreshes once on 401 and retries with the new token', () => {
    auth.login('a@example.com', 'pw').subscribe();
    ctrl.expectOne('/api/v1/auth/login/').flush({ access: 'old', refresh: 'r1' });
    let body: unknown;
    http.get('/api/v1/profile/').subscribe(b => (body = b));
    const first = ctrl.expectOne('/api/v1/profile/');
    expect(first.request.headers.get('Authorization')).toBe('Bearer old');
    first.flush({}, { status: 401, statusText: 'Unauthorized' });
    ctrl.expectOne('/api/v1/auth/refresh/').flush({ access: 'new', refresh: 'r2' });
    const retry = ctrl.expectOne('/api/v1/profile/');
    expect(retry.request.headers.get('Authorization')).toBe('Bearer new');
    retry.flush({ ok: true });
    expect(body).toEqual({ ok: true });
    expect(localStorage.getItem('jobpilot.refresh')).toBe('r2');
  });

  it('never sends the bearer token to the login endpoint', () => {
    auth.login('a@example.com', 'pw').subscribe();
    const req = ctrl.expectOne('/api/v1/auth/login/');
    expect(req.request.headers.has('Authorization')).toBeFalse();
    req.flush({ access: 'a', refresh: 'r' });
  });
});
