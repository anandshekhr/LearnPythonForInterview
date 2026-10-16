import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { Router } from '@angular/router';
import { AuthService } from '../core/auth.service';
import { errorMessage } from '../core/notify.service';

@Component({
  standalone: true,
  imports: [ReactiveFormsModule, MatCardModule, MatFormFieldModule, MatInputModule, MatButtonModule, MatIconModule],
  template: `
  <div class="auth-page">
    <mat-card class="auth-card">
      <div class="brand big"><mat-icon>rocket_launch</mat-icon> JobPilot</div>
      <p class="muted">Quality-first job discovery and applications.</p>
      <h2>{{ mode() === 'login' ? 'Sign in' : 'Create your account' }}</h2>
      <form [formGroup]="form" (ngSubmit)="submit()">
        @if (mode() === 'register') {
          <mat-form-field appearance="outline"><mat-label>Full name</mat-label><input matInput formControlName="full_name" autocomplete="name"></mat-form-field>
        }
        <mat-form-field appearance="outline"><mat-label>Email</mat-label>
          <input matInput type="email" formControlName="email" autocomplete="email" required></mat-form-field>
        <mat-form-field appearance="outline"><mat-label>Password</mat-label>
          <input matInput type="password" formControlName="password" [autocomplete]="mode() === 'login' ? 'current-password' : 'new-password'" required>
          @if (mode() === 'register') { <mat-hint>At least 10 characters, not a common password.</mat-hint> }
        </mat-form-field>
        @if (error()) { <p class="error-text" role="alert">{{ error() }}</p> }
        <button mat-flat-button color="primary" class="full" [disabled]="form.invalid || busy()">
          {{ busy() ? 'Please wait…' : (mode() === 'login' ? 'Sign in' : 'Create account') }}</button>
      </form>
      <button mat-button class="full" (click)="toggle()">
        {{ mode() === 'login' ? "New here? Create an account" : 'Already registered? Sign in' }}</button>
    </mat-card>
  </div>`,
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);
  mode = signal<'login' | 'register'>('login');
  busy = signal(false);
  error = signal('');
  form = inject(FormBuilder).nonNullable.group({
    full_name: [''], email: ['', [Validators.required, Validators.email]], password: ['', [Validators.required, Validators.minLength(8)]],
  });

  toggle(): void { this.mode.set(this.mode() === 'login' ? 'register' : 'login'); this.error.set(''); }

  submit(): void {
    if (this.form.invalid) return;
    this.busy.set(true);
    this.error.set('');
    const { email, password, full_name } = this.form.getRawValue();
    const req = this.mode() === 'login' ? this.auth.login(email, password) : this.auth.register(email, password, full_name);
    req.subscribe({
      next: () => this.router.navigate([this.mode() === 'login' ? '/dashboard' : '/resumes']),
      error: e => { this.busy.set(false); this.error.set(errorMessage(e, 'Sign-in failed.')); },
    });
  }
}
