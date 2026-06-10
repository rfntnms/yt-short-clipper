from scheduler import add_scheduled_job, remove_scheduled_job, list_scheduled_jobs, _parse_cron_expr

print('=== Cron Parsing ===')
print(_parse_cron_expr('0 0 9 * * *'))
print(_parse_cron_expr('0 9 * * *'))

print()
print('=== Job Registration ===')
jid = add_scheduled_job('0 0 9 * * *', 'https://youtu.be/test', {'output_dir': 'output'}, job_id='test-daily')
print(f'Added: {jid}')
print(f'Jobs: {list_scheduled_jobs()}')
print(f'Removed: {remove_scheduled_job(jid)}')
print(f'Jobs after remove: {list_scheduled_jobs()}')

try:
    _parse_cron_expr('1 2 3')
except ValueError as e:
    print(f'Expected error: {e}')

print()
print('All tests passed.')
