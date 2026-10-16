import { HttpErrorResponse } from '@angular/common/http';
import { errorMessage } from './notify.service';

describe('errorMessage', () => {
  it('uses the API error envelope and readiness problems', () => {
    const err = new HttpErrorResponse({ status: 400, error: { error: { code: 'answers_incomplete', message: 'Some answers need your attention.',
      details: { problems: ['Required question unanswered: Email'] } } } });
    expect(errorMessage(err)).toBe('Some answers need your attention. Required question unanswered: Email');
  });
  it('flattens DRF field errors', () => {
    const err = new HttpErrorResponse({ status: 400, error: { email: ['Enter a valid email.'], request_id: 'x' } });
    expect(errorMessage(err)).toBe('email: Enter a valid email.');
  });
  it('explains network failures', () => {
    expect(errorMessage(new HttpErrorResponse({ status: 0 }))).toContain('Cannot reach the server');
  });
});
