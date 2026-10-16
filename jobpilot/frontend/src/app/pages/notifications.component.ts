import { Component, OnInit, inject, signal } from '@angular/core';
import { PageEvent } from '@angular/material/paginator';
import { Router } from '@angular/router';
import { ApiService } from '../core/api.service';
import { AppNotification, Paginated } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

const ICONS: Record<string, string> = {
  high_match: 'star', awaiting_review: 'rate_review', submitted: 'send', submission_failed: 'error', action_required: 'priority_high',
  interview: 'event', follow_up: 'schedule', job_expired: 'event_busy', source_error: 'link_off', system: 'info',
};

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Notifications">
    <mat-slide-toggle [checked]="unreadOnly()" (change)="unreadOnly.set($event.checked); load()">Unread only</mat-slide-toggle>
    <button mat-stroked-button (click)="readAll()"><mat-icon>done_all</mat-icon> Mark all read</button>
  </jp-page-header>
  <jp-loading [loading]="loading()" />
  @for (n of items(); track n.id) {
    <div class="row-link card-like" [class.unread]="!n.read_at" (click)="open(n)" tabindex="0" (keydown.enter)="open(n)">
      <mat-icon>{{ icon(n.kind) }}</mat-icon>
      <div class="grow"><div class="strong">{{ n.title }}</div>@if (n.body) { <div class="muted small clamp">{{ n.body }}</div> }</div>
      <span class="muted small">{{ n.created_at | date: 'short' }}</span>
    </div>
  } @empty { @if (!loading()) { <jp-empty icon="notifications_none" title="No notifications" /> } }
  <mat-paginator [length]="count()" [pageSize]="20" (page)="page($event)" />`,
})
export class NotificationsComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  private router = inject(Router);
  loading = signal(true);
  items = signal<AppNotification[]>([]);
  count = signal(0);
  unreadOnly = signal(false);
  private pageNo = 1;

  ngOnInit(): void { this.load(); }
  icon(kind: string): string { return ICONS[kind] ?? 'notifications'; }
  load(): void {
    this.api.get<Paginated<AppNotification>>('notifications/', { unread: this.unreadOnly() ? 1 : '', page: this.pageNo }).subscribe({
      next: r => { this.items.set(r.results); this.count.set(r.count); this.loading.set(false); },
      error: e => { this.loading.set(false); this.notify.error(e); } });
  }
  page(e: PageEvent): void { this.pageNo = e.pageIndex + 1; this.load(); }
  open(n: AppNotification): void {
    if (!n.read_at) this.api.post(`notifications/${n.id}/read/`).subscribe();
    if (n.link) this.router.navigateByUrl(n.link);
  }
  readAll(): void { this.api.post('notifications/read-all/').subscribe({ next: () => this.load() }); }
}
