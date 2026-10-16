import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ApiService } from '../core/api.service';
import { Interview, Paginated } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { EmptyStateComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

interface Day { date: Date; inMonth: boolean; items: Interview[]; }

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, EmptyStateComponent],
  template: `
  <jp-page-header title="Interview calendar" subtitle="Interviews are added from an application's page.">
    <button mat-icon-button (click)="shift(-1)" aria-label="Previous month"><mat-icon>chevron_left</mat-icon></button>
    <strong>{{ month() | date: 'MMMM y' }}</strong>
    <button mat-icon-button (click)="shift(1)" aria-label="Next month"><mat-icon>chevron_right</mat-icon></button>
    <button mat-button (click)="month.set(startOfMonth(today)); load()">Today</button>
  </jp-page-header>
  <jp-loading [loading]="loading()" />
  <div class="calendar" role="grid" aria-label="Month view">
    @for (d of weekdays; track d) { <div class="cal-head" role="columnheader">{{ d }}</div> }
    @for (day of days(); track day.date.getTime()) {
      <div class="cal-cell" role="gridcell" [class.out]="!day.inMonth" [class.today]="isToday(day.date)">
        <div class="cal-date">{{ day.date.getDate() }}</div>
        @for (i of day.items; track i.id) {
          <a class="cal-item" [class]="'outcome-' + i.outcome" [routerLink]="['/applications', i.application]" [matTooltip]="i.company_name + ' · ' + i.round_name">
            {{ i.scheduled_at | date: 'HH:mm' }} {{ i.company_name }}</a>
        }
      </div>
    }
  </div>
  <h2 class="section-title">Upcoming</h2>
  @for (i of upcoming(); track i.id) {
    <a class="row-link card-like" [routerLink]="['/applications', i.application]"><mat-icon>event</mat-icon>
      <div class="grow"><div class="strong">{{ i.company_name }} — {{ i.round_name }}</div>
        <div class="muted">{{ i.job_title }} · {{ i.scheduled_at | date: 'EEE d MMM, HH:mm' }} · {{ i.duration_minutes }} min · {{ i.mode }}</div></div>
      @if (i.location) { <span class="muted small">{{ i.location }}</span> }</a>
  } @empty { <jp-empty icon="event_available" title="No upcoming interviews" /> }`,
})
export class InterviewsComponent implements OnInit {
  private api = inject(ApiService);
  private notify = inject(NotifyService);
  today = new Date();
  weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  loading = signal(true);
  month = signal(this.startOfMonth(new Date()));
  interviews = signal<Interview[]>([]);
  upcoming = signal<Interview[]>([]);
  days = computed<Day[]>(() => {
    const first = this.month();
    const start = new Date(first);
    start.setDate(1 - ((first.getDay() + 6) % 7));
    return Array.from({ length: 42 }, (_, n) => {
      const date = new Date(start);
      date.setDate(start.getDate() + n);
      const items = this.interviews().filter(i => new Date(i.scheduled_at).toDateString() === date.toDateString());
      return { date, inMonth: date.getMonth() === first.getMonth(), items };
    });
  });

  ngOnInit(): void {
    this.load();
    this.api.get<Paginated<Interview>>('interviews/', { start: new Date().toISOString(), outcome: 'pending', page_size: 20 })
      .subscribe(r => this.upcoming.set(r.results));
  }
  startOfMonth(d: Date): Date { return new Date(d.getFullYear(), d.getMonth(), 1); }
  isToday(d: Date): boolean { return d.toDateString() === this.today.toDateString(); }
  shift(n: number): void { const m = this.month(); this.month.set(new Date(m.getFullYear(), m.getMonth() + n, 1)); this.load(); }

  load(): void {
    const days = this.days();
    this.loading.set(true);
    this.api.get<Paginated<Interview>>('interviews/', { start: days[0].date.toISOString(), end: days[41].date.toISOString(), page_size: 100 })
      .subscribe({ next: r => { this.interviews.set(r.results); this.loading.set(false); }, error: e => { this.loading.set(false); this.notify.error(e); } });
  }
}
