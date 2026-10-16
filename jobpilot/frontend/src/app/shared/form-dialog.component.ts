import { Component, inject } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';

export interface FieldDef {
  key: string; label: string; type?: 'text' | 'textarea' | 'date' | 'datetime-local' | 'number' | 'checkbox' | 'list' | 'select' | 'url' | 'email';
  required?: boolean; hint?: string; options?: { value: string; label: string }[];
}
export interface FormDialogData { title: string; fields: FieldDef[]; value?: Record<string, any>; }

/** Generic create/edit dialog. `list` fields are edited one item per line. */
@Component({
  standalone: true,
  imports: [ReactiveFormsModule, MatDialogModule, MatButtonModule, MatFormFieldModule, MatInputModule, MatCheckboxModule, MatSelectModule],
  template: `
  <h2 mat-dialog-title>{{ data.title }}</h2>
  <mat-dialog-content>
    <form [formGroup]="form" class="dialog-form" (ngSubmit)="save()">
      @for (f of data.fields; track f.key) {
        @switch (f.type) {
          @case ('checkbox') { <mat-checkbox [formControlName]="f.key">{{ f.label }}</mat-checkbox> }
          @case ('textarea') {
            <mat-form-field appearance="outline"><mat-label>{{ f.label }}</mat-label>
              <textarea matInput rows="4" [formControlName]="f.key"></textarea>@if (f.hint) {<mat-hint>{{ f.hint }}</mat-hint>}</mat-form-field>
          }
          @case ('list') {
            <mat-form-field appearance="outline"><mat-label>{{ f.label }}</mat-label>
              <textarea matInput rows="5" [formControlName]="f.key"></textarea><mat-hint>One per line</mat-hint></mat-form-field>
          }
          @case ('select') {
            <mat-form-field appearance="outline"><mat-label>{{ f.label }}</mat-label>
              <mat-select [formControlName]="f.key">@for (o of f.options; track o.value) {<mat-option [value]="o.value">{{ o.label }}</mat-option>}</mat-select>
            </mat-form-field>
          }
          @default {
            <mat-form-field appearance="outline"><mat-label>{{ f.label }}</mat-label>
              <input matInput [type]="f.type || 'text'" [formControlName]="f.key">@if (f.hint) {<mat-hint>{{ f.hint }}</mat-hint>}
              @if (form.get(f.key)?.hasError('required')) {<mat-error>Required</mat-error>}</mat-form-field>
          }
        }
      }
    </form>
  </mat-dialog-content>
  <mat-dialog-actions align="end">
    <button mat-button mat-dialog-close type="button">Cancel</button>
    <button mat-flat-button color="primary" (click)="save()" [disabled]="form.invalid">Save</button>
  </mat-dialog-actions>`,
})
export class FormDialogComponent {
  data = inject<FormDialogData>(MAT_DIALOG_DATA);
  private ref = inject(MatDialogRef<FormDialogComponent>);
  form = new FormGroup<Record<string, FormControl>>({});

  constructor() {
    for (const f of this.data.fields) {
      let value = this.data.value?.[f.key] ?? (f.type === 'checkbox' ? false : '');
      if (f.type === 'list' && Array.isArray(value)) value = value.join('\n');
      this.form.addControl(f.key, new FormControl(value, f.required ? Validators.required : []));
    }
  }

  save(): void {
    if (this.form.invalid) return;
    const out: Record<string, any> = {};
    for (const f of this.data.fields) {
      let v = this.form.value[f.key];
      if (f.type === 'list') v = String(v || '').split('\n').map(s => s.trim()).filter(Boolean);
      if ((f.type === 'date' || f.type === 'datetime-local' || f.type === 'number') && v === '') v = null;
      out[f.key] = v;
    }
    this.ref.close(out);
  }
}
