import { TestBed } from '@angular/core/testing';
import { ScoreBadgeComponent, StatusChipComponent, salaryRange } from './ui';

describe('shared ui', () => {
  it('buckets scores when no category is given', () => {
    const f = TestBed.createComponent(ScoreBadgeComponent);
    f.componentInstance.score = 72;
    f.detectChanges();
    const el: HTMLElement = f.nativeElement.querySelector('.score');
    expect(el.textContent?.trim()).toBe('72');
    expect(el.className).toContain('score-good');
  });
  it('labels statuses', () => {
    const f = TestBed.createComponent(StatusChipComponent);
    f.componentInstance.status = 'interview_scheduled';
    f.detectChanges();
    expect(f.nativeElement.textContent.trim()).toBe('Interview scheduled');
  });
  it('formats INR salary ranges in LPA', () => {
    expect(salaryRange(3000000, 4000000, 'INR')).toBe('₹30.0 – ₹40.0 LPA');
    expect(salaryRange(null, null, 'INR')).toBe('');
  });
});
