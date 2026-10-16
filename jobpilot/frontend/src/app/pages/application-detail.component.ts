import { Component, OnInit, inject, input, signal } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { forkJoin } from 'rxjs';
import { ApiService } from '../core/api.service';
import { Answer, Application, ApplicationKit, GeneratedDocument, Paginated, ResumeVersion, STATUS_LABELS } from '../core/models';
import { NotifyService } from '../core/notify.service';
import { FormDialogComponent } from '../shared/form-dialog.component';
import { MATERIAL } from '../shared/material';
import { ConfirmDialogComponent, LoadingComponent, PageHeaderComponent, ScoreBadgeComponent, StatusChipComponent } from '../shared/ui';

@Component({
  standalone: true,
  imports: [...MATERIAL, PageHeaderComponent, LoadingComponent, StatusChipComponent, ScoreBadgeComponent],
  template: `
  <jp-loading [loading]="!app()" />
  @if (app(); as a) {
    <jp-page-header [title]="a.job_title" [subtitle]="a.company_name">
      <jp-status [status]="a.status" />
      @if (a.job) { <a mat-button [routerLink]="['/jobs', a.job]"><mat-icon>work</mat-icon> Job & match</a> }
    </jp-page-header>
    <div class="grid-2 wide-left">
      <div>
        <mat-card class="item-card">
          <h3>Apply</h3>
          @if (kit(); as k) {
            @if (!k.job_active) { <p class="error-text">This posting is no longer active.</p> }
            <ol class="steps">
              <li [class.done]="!!a.resume_version">
                <strong>Approved resume.</strong>
                @if (a.resume_version) { {{ a.resume_version_label }} @if (a.cover_letter) { + cover letter } }
                @else {
                  <span class="muted">Tailor and approve a resume, or attach an approved version:</span>
                  <div class="row">
                    <mat-form-field appearance="outline" class="grow"><mat-label>Approved resume version</mat-label>
                      <mat-select [(ngModel)]="selectedVersion">@for (v of approvedVersions(); track v.id) { <mat-option [value]="v.id">v{{ v.version_number }} · {{ v.label }}</mat-option> }</mat-select></mat-form-field>
                    <mat-form-field appearance="outline" class="grow"><mat-label>Approved cover letter (optional)</mat-label>
                      <mat-select [(ngModel)]="selectedLetter"><mat-option [value]="null">None</mat-option>
                        @for (l of approvedLetters(); track l.id) { <mat-option [value]="l.id">{{ l.title }}</mat-option> }</mat-select></mat-form-field>
                    <button mat-stroked-button (click)="attach()" [disabled]="!selectedVersion">Attach</button>
                  </div>
                }
              </li>
              <li [class.done]="a.answers?.length && !k.problems.length">
                <strong>Application questions.</strong>
                <button mat-button (click)="prepareAnswers()"><mat-icon>quiz</mat-icon> {{ a.answers?.length ? 'Refresh' : 'Load' }} questions</button>
                @if (answers().length) {
                  @for (q of answers(); track q.id) {
                    @if (q.field_type !== 'file') {
                      <div class="answer" [class.sensitive]="q.sensitive">
                        <label>{{ q.question_label }} @if (q.required) {<span class="req">*</span>} @if (q.sensitive) {<span class="badge warn">sensitive – your answer only</span>}</label>
                        @if (q.options.length) {
                          <mat-form-field appearance="outline" class="full"><mat-select [(ngModel)]="q.answer">
                            <mat-option value="">—</mat-option>@for (o of q.options; track o.value) { <mat-option [value]="'' + o.value">{{ o.label }}</mat-option> }</mat-select></mat-form-field>
                        } @else {
                          <mat-form-field appearance="outline" class="full"><input matInput [(ngModel)]="q.answer"></mat-form-field>
                        }
                        @if (q.sensitive) { <mat-checkbox [(ngModel)]="reviewed[q.id]">I have reviewed this answer and want it used</mat-checkbox> }
                      </div>
                    }
                  }
                  <button mat-stroked-button (click)="saveAnswers()" [disabled]="a.status === 'submitted'">Save answers</button>
                }
                @for (p of k.problems; track p) { <div class="error-text small">{{ p }}</div> }
              </li>
              <li [class.done]="a.status === 'submitted' || !!a.submitted_at">
                <strong>Submit.</strong>
                @if (k.automated_submission.available) {
                  <button mat-flat-button color="primary" (click)="submitApi(false)" [disabled]="!canSubmit(a)"><mat-icon>send</mat-icon> Submit via authorized API</button>
                  @if (a.submission_response?.unconfirmed) {
                    <button mat-stroked-button color="warn" (click)="submitApi(true)">I checked – retry submission</button>
                  }
                } @else {
                  <div class="muted small">Automated submission unavailable: {{ k.automated_submission.reason || 'enable it in Automation settings' }}</div>
                }
                <div class="row wrap">
                  <a mat-flat-button color="accent" [href]="k.official_url" target="_blank" rel="noopener noreferrer"><mat-icon>open_in_new</mat-icon> Open official application page</a>
                  @if (a.resume_document) { <button mat-stroked-button (click)="api.download('documents/' + a.resume_document + '/download/', 'resume.pdf')"><mat-icon>download</mat-icon> Resume</button> }
                  <button mat-stroked-button (click)="confirmSubmitted()" [disabled]="!canSubmit(a)"><mat-icon>task_alt</mat-icon> I submitted it</button>
                </div>
                @if (k.browser_assist_enabled) {
                  <div class="muted small">Want form pre-filling? Run <code>python manage.py assist_apply {{ a.id }}</code> on your computer; it fills the official form and leaves the final review and submit to you.</div>
                }
                @if (a.last_error) { <div class="error-text">{{ a.last_error }}</div> }
              </li>
            </ol>
          }
        </mat-card>
        <mat-card class="item-card">
          <h3>Notes & follow-up</h3>
          <mat-form-field appearance="outline" class="full"><mat-label>Recruiter notes</mat-label><textarea matInput rows="4" [(ngModel)]="a.notes"></textarea></mat-form-field>
          <mat-form-field appearance="outline"><mat-label>Follow-up date</mat-label><input matInput type="date" [(ngModel)]="a.follow_up_date"></mat-form-field>
          <button mat-stroked-button (click)="saveNotes()">Save</button>
        </mat-card>
      </div>
      <div>
        <mat-card class="item-card">
          <h3>Status</h3>
          <div class="muted small">Match <jp-score [score]="a.match_score" /> · mode {{ a.mode }} · created {{ a.created_at | date }}</div>
          @if (a.submitted_at) { <p>Submitted {{ a.submitted_at | date: 'medium' }} via {{ a.submission_channel }} {{ a.submission_reference ? '· ref ' + a.submission_reference : '' }}</p> }
          @if (a.allowed_transitions?.length) {
            <mat-form-field appearance="outline" class="full"><mat-label>Update status</mat-label>
              <mat-select (selectionChange)="changeStatus($event.value)">
                @for (s of a.allowed_transitions; track s) { <mat-option [value]="s">{{ label(s) }}</mat-option> }</mat-select></mat-form-field>
          }
          <h4>History</h4>
          <ul class="timeline">
            @for (h of a.history; track h.id) {
              <li><strong>{{ label(h.to_status) }}</strong> <span class="muted small">{{ h.created_at | date: 'medium' }} · {{ h.actor }}</span>
                @if (h.note) { <div class="small">{{ h.note }}</div> }</li>
            }
          </ul>
        </mat-card>
        <mat-card class="item-card">
          <div class="row"><h3 class="grow">Interviews</h3><button mat-stroked-button (click)="addInterview()"><mat-icon>add</mat-icon> Add</button></div>
          @for (i of a.interviews; track i.id) {
            <div class="row-link"><mat-icon>event</mat-icon><div class="grow"><div class="strong">{{ i.round_name }}</div>
              <div class="muted small">{{ i.scheduled_at | date: 'EEE d MMM y, HH:mm' }} · {{ i.mode }} · {{ i.outcome }}</div></div></div>
          } @empty { <p class="muted">None yet.</p> }
        </mat-card>
        <mat-card class="item-card">
          <div class="row"><h3 class="grow">Follow-ups</h3><button mat-stroked-button (click)="addFollowUp()"><mat-icon>add</mat-icon> Add</button></div>
          @for (f of a.follow_ups; track f.id) {
            <div class="row-link"><mat-checkbox [checked]="!!f.completed_at" (change)="completeFollowUp(f.id, $event.checked)"></mat-checkbox>
              <div class="grow">{{ f.due_date | date }} — {{ f.note }}</div></div>
          } @empty { <p class="muted">None yet.</p> }
        </mat-card>
        <button mat-button color="warn" (click)="remove()" [disabled]="!!a.submitted_at"><mat-icon>delete</mat-icon> Delete application</button>
      </div>
    </div>
  }`,
})
export class ApplicationDetailComponent implements OnInit {
  id = input.required<string>();
  api = inject(ApiService);
  private notify = inject(NotifyService);
  private dialog = inject(MatDialog);
  app = signal<Application | null>(null);
  kit = signal<ApplicationKit | null>(null);
  answers = signal<Answer[]>([]);
  approvedVersions = signal<ResumeVersion[]>([]);
  approvedLetters = signal<GeneratedDocument[]>([]);
  reviewed: Record<number, boolean> = {};
  selectedVersion: number | null = null;
  selectedLetter: number | null = null;

  ngOnInit(): void { this.load(); }

  load(): void {
    forkJoin({ app: this.api.get<Application>(`applications/${this.id()}/`), kit: this.api.get<ApplicationKit>(`applications/${this.id()}/kit/`) })
      .subscribe({
        next: r => {
          this.app.set(r.app); this.kit.set(r.kit);
          this.answers.set(structuredClone(r.app.answers ?? []));
          this.reviewed = Object.fromEntries((r.app.answers ?? []).filter(x => x.sensitive).map(x => [x.id, !!x.reviewed_at]));
          if (!r.app.resume_version) {
            this.api.get<Paginated<ResumeVersion>>('resume-versions/', { status: 'approved', page_size: 50 }).subscribe(v => this.approvedVersions.set(v.results));
            this.api.get<Paginated<GeneratedDocument>>('documents/', { kind: 'cover_letter', status: 'approved' }).subscribe(d => this.approvedLetters.set(d.results));
          }
        },
        error: e => this.notify.error(e, 'Application not found.'),
      });
  }

  label(s: string): string { return STATUS_LABELS[s] ?? s; }
  canSubmit(a: Application): boolean { return ['ready_to_apply', 'submission_failed'].includes(a.status); }

  attach(): void {
    this.api.post(`applications/${this.id()}/attach-resume/`, { resume_version: this.selectedVersion, cover_letter: this.selectedLetter }).subscribe({
      next: () => { this.notify.success('Resume attached. Ready to apply.'); this.load(); }, error: e => this.notify.error(e) });
  }
  prepareAnswers(): void {
    this.api.post(`applications/${this.id()}/prepare-answers/`).subscribe({ next: () => setTimeout(() => this.load(), 800), error: e => this.notify.error(e) });
  }
  saveAnswers(): void {
    const body = this.answers().filter(q => q.field_type !== 'file').map(q => ({ id: q.id, answer: q.answer ?? '', reviewed: !!this.reviewed[q.id] }));
    this.api.patch(`applications/${this.id()}/answers/`, body).subscribe({ next: () => { this.notify.success('Answers saved.'); this.load(); }, error: e => this.notify.error(e) });
  }

  submitApi(confirmRetry: boolean): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Submit application?', confirm: 'Submit',
      message: 'Your approved resume and reviewed answers will be sent to the employer through their authorized API. This cannot be undone.' } })
      .afterClosed().subscribe(ok => {
        if (!ok) return;
        const key = crypto.randomUUID();
        this.api.post(`applications/${this.id()}/submit/`, { confirm_retry: confirmRetry }, { 'Idempotency-Key': key }).subscribe({
          next: () => { this.notify.success('Submission sent. The status updates when the employer confirms.'); this.load(); },
          error: e => { this.notify.error(e); this.load(); } });
      });
  }

  confirmSubmitted(): void {
    this.dialog.open(FormDialogComponent, { width: '480px', data: { title: 'Record your submission', fields: [
      { key: 'reference', label: 'Confirmation number / reference (optional)' },
      { key: 'channel', label: 'How did you submit?', type: 'select', options: [
        { value: 'manual', label: 'On the official page' }, { value: 'browser_assisted', label: 'With the browser assistant' }] },
    ], value: { channel: 'manual' } } }).afterClosed().subscribe(v => v && this.api.post(`applications/${this.id()}/confirm-submitted/`, v).subscribe({
      next: () => { this.notify.success('Recorded as submitted.'); this.load(); }, error: e => this.notify.error(e) }));
  }

  changeStatus(status: string): void {
    this.api.post(`applications/${this.id()}/status/`, { status }).subscribe({ next: () => { this.notify.success('Status updated.'); this.load(); },
      error: e => { this.notify.error(e); this.load(); } });
  }
  saveNotes(): void {
    const a = this.app()!;
    this.api.patch(`applications/${this.id()}/`, { notes: a.notes, follow_up_date: a.follow_up_date || null }).subscribe({
      next: () => this.notify.success('Saved.'), error: e => this.notify.error(e) });
  }

  addInterview(): void {
    this.dialog.open(FormDialogComponent, { width: '560px', data: { title: 'Add interview', fields: [
      { key: 'round_name', label: 'Round', required: true }, { key: 'scheduled_at', label: 'Date & time', type: 'datetime-local', required: true },
      { key: 'duration_minutes', label: 'Duration (minutes)', type: 'number' },
      { key: 'mode', label: 'Mode', type: 'select', options: [{ value: 'video', label: 'Video' }, { value: 'phone', label: 'Phone' }, { value: 'onsite', label: 'On-site' }] },
      { key: 'location', label: 'Link or address' }, { key: 'interviewers', label: 'Interviewers' }, { key: 'notes', label: 'Notes', type: 'textarea' },
    ], value: { mode: 'video', duration_minutes: 60 } } }).afterClosed().subscribe(v => {
      if (!v) return;
      v.scheduled_at = new Date(v.scheduled_at).toISOString();
      this.api.post('interviews/', { ...v, application: Number(this.id()) }).subscribe({ next: () => this.load(), error: e => this.notify.error(e) });
    });
  }
  addFollowUp(): void {
    this.dialog.open(FormDialogComponent, { width: '480px', data: { title: 'Add follow-up', fields: [
      { key: 'due_date', label: 'Due date', type: 'date', required: true }, { key: 'note', label: 'Note' }] } })
      .afterClosed().subscribe(v => v && this.api.post('follow-ups/', { ...v, application: Number(this.id()) }).subscribe({ next: () => this.load(), error: e => this.notify.error(e) }));
  }
  completeFollowUp(id: number, done: boolean): void {
    this.api.patch(`follow-ups/${id}/`, { completed_at: done ? new Date().toISOString() : null }).subscribe({ error: e => this.notify.error(e) });
  }
  remove(): void {
    this.dialog.open(ConfirmDialogComponent, { data: { title: 'Delete application?', message: 'Its history will be removed.', confirm: 'Delete', danger: true } })
      .afterClosed().subscribe(ok => ok && this.api.delete(`applications/${this.id()}/`).subscribe({
        next: () => history.back(), error: e => this.notify.error(e) }));
  }
}
