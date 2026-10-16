import { DecimalPipe } from '@angular/common';
import { Component, ElementRef, Input, OnChanges, OnDestroy, ViewChild, inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import { STATUS_LABELS } from '../core/models';

Chart.register(...registerables);

@Component({
  selector: 'jp-score',
  standalone: true,
  template: `<span class="score score-{{ category || bucket() }}" [attr.aria-label]="'Match score ' + score">{{ score ?? '–' }}</span>`,
})
export class ScoreBadgeComponent {
  @Input() score: number | null | undefined;
  @Input() category = '';
  bucket(): string {
    const s = this.score ?? 0;
    return s >= 80 ? 'excellent' : s >= 65 ? 'good' : s >= 45 ? 'possible' : 'low';
  }
}

@Component({
  selector: 'jp-status',
  standalone: true,
  template: `<span class="status-chip status-{{ status }}">{{ label() }}</span>`,
})
export class StatusChipComponent {
  @Input() status = '';
  label(): string { return STATUS_LABELS[this.status] ?? this.status.replace(/_/g, ' '); }
}

@Component({
  selector: 'jp-empty',
  standalone: true,
  imports: [MatIconModule],
  template: `<div class="empty"><mat-icon>{{ icon }}</mat-icon><h3>{{ title }}</h3><p>{{ text }}</p><ng-content /></div>`,
})
export class EmptyStateComponent {
  @Input() icon = 'inbox';
  @Input() title = 'Nothing here yet';
  @Input() text = '';
}

@Component({
  selector: 'jp-loading',
  standalone: true,
  imports: [MatProgressBarModule],
  template: `@if (loading) { <mat-progress-bar mode="indeterminate" /> }`,
})
export class LoadingComponent {
  @Input() loading = false;
}

@Component({
  selector: 'jp-page-header',
  standalone: true,
  template: `<header class="page-header"><div><h1>{{ title }}</h1>@if (subtitle) {<p>{{ subtitle }}</p>}</div>
    <div class="actions"><ng-content /></div></header>`,
})
export class PageHeaderComponent {
  @Input() title = '';
  @Input() subtitle = '';
}

export interface ConfirmData { title: string; message: string; confirm?: string; danger?: boolean; }

@Component({
  standalone: true,
  imports: [MatDialogModule, MatButtonModule],
  template: `<h2 mat-dialog-title>{{ data.title }}</h2>
    <mat-dialog-content><p class="pre">{{ data.message }}</p></mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button [mat-dialog-close]="false">Cancel</button>
      <button mat-flat-button [color]="data.danger ? 'warn' : 'primary'" [mat-dialog-close]="true">{{ data.confirm || 'Confirm' }}</button>
    </mat-dialog-actions>`,
})
export class ConfirmDialogComponent {
  data = inject<ConfirmData>(MAT_DIALOG_DATA);
}

@Component({
  selector: 'jp-chart',
  standalone: true,
  template: `<div class="chart-box" [style.height.px]="height"><canvas #canvas role="img" [attr.aria-label]="label"></canvas></div>`,
})
export class ChartComponent implements OnChanges, OnDestroy {
  @Input({ required: true }) config!: ChartConfiguration;
  @Input() height = 260;
  @Input() label = 'Chart';
  @ViewChild('canvas', { static: true }) canvas!: ElementRef<HTMLCanvasElement>;
  private chart?: Chart;

  ngOnChanges(): void {
    this.chart?.destroy();
    if (this.config) this.chart = new Chart(this.canvas.nativeElement, this.config);
  }
  ngOnDestroy(): void { this.chart?.destroy(); }
}

export function money(amount: number | null | undefined, currency = 'INR'): string {
  if (amount == null) return '';
  if (currency === 'INR' && amount >= 100000) return `₹${(amount / 100000).toFixed(1)} LPA`;
  return `${currency} ${amount.toLocaleString()}`;
}

export function salaryRange(min: number | null, max: number | null, currency: string): string {
  if (!min && !max) return '';
  if (min && max) return `${money(min, currency)} – ${money(max, currency)}`.replace(' LPA –', ' –');
  return money(min || max, currency);
}

export const SHARED = [ScoreBadgeComponent, StatusChipComponent, EmptyStateComponent, LoadingComponent, PageHeaderComponent, DecimalPipe];
