import { Component, OnDestroy, OnInit, inject, input, signal } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Comparison, GeneratedDocument, Paginated, ResumeContent, ResumeVersion } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, LoadingComponent, PageHeaderComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent],
  template: `
  <jp-loading [loading]="!version()" />
  @if (version(); as v) {
    <jp-page-header [title]="v.label" [subtitle]="'Version ' + v.version_number + ' · ' + v.kind + ' · ' + v.status.replace('_', ' ')">
      @if (v.job) { <a mat-button [routerLink]="['/jobs', v.job]"><mat-icon>work</mat-icon> Job</a> }
      @if (editable()) {
        <button mat-stroked-button color="warn" (click)="reject()"><mat-icon>thumb_down</mat-icon> Reject</button>
        <button mat-flat-button color="primary" (click)="approve()"><mat-icon>verified</mat-icon> Approve for applications</button>
      }
    </jp-page-header>
    @if (v.tailoring_notes.warnings?.length) {
      <mat-card class="callout warn"><mat-icon>warning</mat-icon><div>@for (w of v.tailoring_notes.warnings; track w) { <div>{{ w }}</div> }</div></mat-card>
    }
    <mat-tab-group animationDuration="0ms">
      <mat-tab label="Side-by-side">
        @if (comparison(); as c) {
          <div class="compare">
            <div class="compare-head"><span>Master (verified profile)</span><span>Tailored for this job</span></div>
            @for (s of c.sections; track s.section) {
              <div class="compare-row" [class.changed]="s.changed">
                <div class="compare-title">{{ s.heading || s.section }} @if (s.changed) { <span class="badge warn">changed</span> }</div>
                <div class="compare-cols">
                  <div>@if (isList(s.master)) { <ul>@for (x of s.master; track $index) { <li>{{ x }}</li> }</ul> } @else { <p>{{ s.master }}</p> }</div>
                  <div>@if (isList(s.tailored)) { <ul>@for (x of s.tailored; track $index) { <li>{{ x }}</li> }</ul> } @else { <p>{{ s.tailored }}</p> }</div>
                </div>
              </div>
            }
          </div>
          @if (c.rewrites.length) {
            <h3>Rephrased by AI (validated against your facts)</h3>
            @for (r of c.rewrites; track $index) { <div class="rewrite"><div class="old">{{ r.original }}</div><div class="new">{{ r.rewritten }}</div></div> }
          }
        }
      </mat-tab>
      <mat-tab label="Requirement map">
        <table class="simple-table">
          <tr><th>Requirement</th><th>Importance</th><th>Status</th><th>Evidence in your profile</th></tr>
          @for (r of v.tailoring_notes.requirement_map ?? []; track r.requirement) {
            <tr><td>{{ r.requirement }}</td><td>{{ r.importance }}</td>
              <td><span class="badge" [class.ok]="r.status === 'matched'" [class.danger]="r.status === 'missing'">{{ r.status.replace('_', ' ') }}</span></td>
              <td>@for (e of r.evidence; track $index) { <div class="small">{{ e.label }}{{ e.text ? ': ' + e.text : '' }}</div> } @empty { <span class="muted">—</span> }</td></tr>
          }
        </table>
      </mat-tab>
      @if (editable() && draft(); as d) {
        <mat-tab label="Edit">
          <p class="muted">You can reorder and reword. Employers, titles, dates, education, certifications and skills must match your verified profile.</p>
          <mat-form-field appearance="outline" class="full"><mat-label>Summary</mat-label><textarea matInput rows="4" [(ngModel)]="d.summary"></textarea></mat-form-field>
          <h4>Skills order</h4>
          <div class="reorder">
            @for (s of d.skills; track s; let i = $index) {
              <span class="chip">{{ s }} <button mat-icon-button (click)="move(d.skills, i, -1)" [disabled]="i === 0" aria-label="Move up"><mat-icon>chevron_left</mat-icon></button></span>
            }
          </div>
          @for (e of d.employment; track e.id) {
            <h4>{{ e.title }} — {{ e.company }}</h4>
            @for (a of e.achievements; track $index; let i = $index) {
              <div class="row">
                <mat-form-field appearance="outline" class="grow"><textarea matInput rows="2" [(ngModel)]="e.achievements[i]"></textarea></mat-form-field>
                <button mat-icon-button (click)="move(e.achievements, i, -1)" [disabled]="i === 0" aria-label="Move up"><mat-icon>arrow_upward</mat-icon></button>
              </div>
            }
          }
          <button mat-flat-button color="primary" (click)="saveEdit()"><mat-icon>save</mat-icon> Save & re-render</button>
        </mat-tab>
      }
      <mat-tab label="Documents ({{ docs().length }})">
        @for (doc of docs(); track doc.id) {
          <mat-card class="item-card">
            <div class="row">
              <mat-icon>{{ doc.kind === 'cover_letter' ? 'mail' : (doc.format === 'pdf' ? 'picture_as_pdf' : 'description') }}</mat-icon>
              <div class="grow"><div class="strong">{{ doc.title }} ({{ doc.format.toUpperCase() }})</div>
                <div class="muted">{{ doc.status.replace('_', ' ') }} · template {{ doc.template }} · {{ doc.generator }}</div></div>
              @if (doc.format === 'pdf') { <button mat-stroked-button (click)="preview(doc)"><mat-icon>visibility</mat-icon> Preview</button> }
              <button mat-icon-button (click)="api.download('documents/' + doc.id + '/download/', doc.kind + '-' + doc.id + '.' + doc.format)" aria-label="Download"><mat-icon>download</mat-icon></button>
            </div>
            @if (doc.kind === 'cover_letter') {
              <mat-form-field appearance="outline" class="full"><mat-label>Cover letter</mat-label>
                <textarea matInput rows="12" [(ngModel)]="doc.text_content" [disabled]="doc.status === 'approved'"></textarea></mat-form-field>
              @if (doc.status !== 'approved') {
                <button mat-stroked-button (click)="saveLetter(doc)">Save letter</button>
                <button mat-flat-button color="primary" (click)="approveLetter(doc)">Approve letter</button>
              }
            }
          </mat-card>
        }
        @if (previewUrl(); as url) { <iframe class="pdf-preview" [src]="url" title="PDF preview"></iframe> }
      </mat-tab>
    </mat-tab-group>
  }`,
})
export class ResumeVersionComponent implements OnInit, OnDestroy {
  id = input.required<string>();
  api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  private sanitizer = inject(DomSanitizer);
  version = signal<ResumeVersion | null>(null);
  comparison = signal<Comparison | null>(null);
  docs = signal<GeneratedDocument[]>([]);
  draft = signal<ResumeContent | null>(null);
  previewUrl = signal<SafeResourceUrl | null>(null);
  private objectUrl = '';

  ngOnInit(): void { this.load(); }
  ngOnDestroy(): void { if (this.objectUrl) URL.revokeObjectURL(this.objectUrl); }

  load(): void {
    forkJoin({
      v: this.api.get<ResumeVersion>(`resume-versions/${this.id()}/`),
      docs: this.api.get<Paginated<GeneratedDocument>>('documents/', { resume_version: this.id() }),
    }).subscribe({
      next: r => {
        this.version.set(r.v); this.docs.set(r.docs.results); this.draft.set(structuredClone(r.v.content));
        if (r.v.kind === 'tailored') this.api.get<Comparison>(`resume-versions/${this.id()}/compare/`).subscribe(c => this.comparison.set(c));
      },
      error: e => this.notify.error(e, 'Resume version not found.'),
    });
  }

  editable(): boolean { const v = this.version(); return !!v && v.kind === 'tailored' && v.status === 'awaiting_approval'; }
  isList(x: unknown): x is unknown[] { return Array.isArray(x); }
  move(list: string[], i: number, d: number): void { const j = i + d; [list[i], list[j]] = [list[j], list[i]]; }

  saveEdit(): void {
    this.api.patch(`resume-versions/${this.id()}/`, { content: this.draft() }).subscribe({
      next: () => { this.notify.success('Saved and re-rendered.'); this.load(); }, error: e => this.notify.error(e) });
  }

  approve(): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Approve this resume?', confirm: 'Approve',
      message: 'I have reviewed this tailored resume and confirm that everything in it is accurate.' } })
      .afterClosed().subscribe(ok => ok && this.api.post(`resume-versions/${this.id()}/approve/`).subscribe({
        next: () => { this.notify.success('Approved. The application is ready to apply.'); this.load(); }, error: e => this.notify.error(e) }));
  }

  reject(): void {
    this.api.post(`resume-versions/${this.id()}/reject/`).subscribe({ next: () => { this.notify.success('Rejected.'); this.load(); }, error: e => this.notify.error(e) });
  }

  preview(doc: GeneratedDocument): void {
    this.api.blob(`documents/${doc.id}/download/`, { inline: 1 }).subscribe(blob => {
      if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = URL.createObjectURL(new Blob([blob], { type: 'application/pdf' }));
      this.previewUrl.set(this.sanitizer.bypassSecurityTrustResourceUrl(this.objectUrl));
    });
  }

  saveLetter(doc: GeneratedDocument): void {
    this.api.patch(`documents/${doc.id}/`, { text_content: doc.text_content }).subscribe({
      next: () => this.notify.success('Cover letter saved.'), error: e => this.notify.error(e) });
  }
  approveLetter(doc: GeneratedDocument): void {
    this.api.patch(`documents/${doc.id}/`, { text_content: doc.text_content }).subscribe({
      next: () => this.api.post(`documents/${doc.id}/approve/`).subscribe({ next: () => { this.notify.success('Cover letter approved.'); this.load(); },
        error: e => this.notify.error(e) }),
      error: e => this.notify.error(e) });
  }
}
