import { BreakpointObserver } from '@angular/cdk/layout';
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { MatBadgeModule } from '@angular/material/badge';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { filter, interval, map, startWith, switchMap } from 'rxjs';
import { ApiService } from '../core/api.service';
import { AuthService } from '../core/auth.service';

interface NavItem { path: string; label: string; icon: string; }

@Component({
  selector: 'jp-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, MatSidenavModule, MatToolbarModule, MatListModule, MatIconModule,
    MatButtonModule, MatBadgeModule, MatMenuModule, MatTooltipModule],
  template: `
  <mat-sidenav-container class="shell">
    <mat-sidenav #nav [mode]="isMobile() ? 'over' : 'side'" [opened]="!isMobile()" class="sidenav">
      <div class="brand"><mat-icon>rocket_launch</mat-icon> JobPilot</div>
      <mat-nav-list>
        @for (group of navGroups; track group.title) {
          <div class="nav-group">{{ group.title }}</div>
          @for (item of group.items; track item.path) {
            <a mat-list-item [routerLink]="item.path" routerLinkActive="active" (click)="isMobile() && nav.close()">
              <mat-icon matListItemIcon>{{ item.icon }}</mat-icon><span matListItemTitle>{{ item.label }}</span>
            </a>
          }
        }
      </mat-nav-list>
    </mat-sidenav>
    <mat-sidenav-content>
      <mat-toolbar class="topbar">
        @if (isMobile()) { <button mat-icon-button (click)="nav.toggle()" aria-label="Menu"><mat-icon>menu</mat-icon></button> }
        <span class="spacer"></span>
        <a mat-icon-button routerLink="/notifications" matTooltip="Notifications" aria-label="Notifications">
          <mat-icon [matBadge]="unread() || null" matBadgeColor="warn" matBadgeSize="small">notifications</mat-icon>
        </a>
        <button mat-button [matMenuTriggerFor]="menu"><mat-icon>account_circle</mat-icon> {{ auth.user()?.full_name || auth.user()?.email }}</button>
        <mat-menu #menu="matMenu">
          <a mat-menu-item routerLink="/profile"><mat-icon>person</mat-icon>Career profile</a>
          <a mat-menu-item routerLink="/settings"><mat-icon>tune</mat-icon>Settings</a>
          <button mat-menu-item (click)="auth.logout()"><mat-icon>logout</mat-icon>Sign out</button>
        </mat-menu>
      </mat-toolbar>
      <main class="content"><router-outlet /></main>
    </mat-sidenav-content>
  </mat-sidenav-container>`,
})
export class ShellComponent implements OnInit {
  auth = inject(AuthService);
  private api = inject(ApiService);
  private router = inject(Router);
  private destroyRef = inject(DestroyRef);
  isMobile = toSignal(inject(BreakpointObserver).observe('(max-width: 959px)').pipe(map(r => r.matches)), { initialValue: false });
  private unreadCount = signal(0);
  unread = computed(() => this.unreadCount());

  navGroups: { title: string; items: NavItem[] }[] = [
    { title: 'Overview', items: [{ path: '/dashboard', label: 'Dashboard', icon: 'dashboard' }, { path: '/analytics', label: 'Analytics', icon: 'insights' }] },
    { title: 'Jobs', items: [
      { path: '/recommended', label: 'Recommended', icon: 'recommend' }, { path: '/jobs', label: 'Job discovery', icon: 'travel_explore' },
      { path: '/sources', label: 'Companies & sources', icon: 'apartment' }] },
    { title: 'Applications', items: [
      { path: '/applications', label: 'Applications', icon: 'assignment' }, { path: '/interviews', label: 'Interview calendar', icon: 'event' },
      { path: '/documents', label: 'Tailored resumes', icon: 'description' }] },
    { title: 'You', items: [
      { path: '/profile', label: 'Career profile', icon: 'badge' }, { path: '/resumes', label: 'Resume manager', icon: 'upload_file' },
      { path: '/settings', label: 'Automation settings', icon: 'tune' }, { path: '/notifications', label: 'Notifications', icon: 'notifications' }] },
  ];

  ngOnInit(): void {
    if (!this.auth.user()) this.auth.loadUser().subscribe({ error: () => undefined });
    const navigations = this.router.events.pipe(filter(e => e instanceof NavigationEnd));
    interval(60000).pipe(startWith(0), switchMap(() => this.api.get<{ count: number }>('notifications/unread-count/')),
      takeUntilDestroyed(this.destroyRef)).subscribe({ next: r => this.unreadCount.set(r.count), error: () => undefined });
    navigations.pipe(switchMap(() => this.api.get<{ count: number }>('notifications/unread-count/')), takeUntilDestroyed(this.destroyRef))
      .subscribe({ next: r => this.unreadCount.set(r.count), error: () => undefined });
  }
}
